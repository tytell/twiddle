import sys, os
import logging
import re
from datetime import datetime
import time
from glob import glob

import nidaqmx.constants as daq
from nidaqmx.stream_writers import CounterWriter, DigitalSingleChannelWriter
from nidaqmx.stream_readers import AnalogMultiChannelReader, DigitalSingleChannelReader
from nidaqmx import Task

from PyQt5 import QtWidgets, QtCore

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import h5py

from twiddleSetup import TwiddleSetupDialog, parameters, SETTINGS_FILE

def set_bit(arr, bit, value):
    return np.bitwise_or(arr, value.astype(np.uint8) * (2**bit))

def first(cond):
    try:
        return np.argwhere(cond)[0][0]
    except IndexError:
        return np.array([])

class Twiddle(object):
    def __init__(self, params, calibration):
        self.params = params
        self.calibration = calibration

        self.errors = []

    def generate_sine(self):
        movement = self.params.child('Movement')

        self.duration = 2*movement['Wait before and after'] + \
                        movement['Cycles']/movement['Frequency']
        movedur = movement['Cycles']/movement['Frequency']

        dt = 1.0/self.params['Motor', 'Pulse frequency']
        self.tout = np.arange(0, self.duration, dt)
        self.tout -= movement['Wait before and after']

        self.vel = 2*np.pi * movement['Frequency'] * movement['Amplitude'] * \
            np.cos(2 * np.pi * movement['Frequency'] * self.tout)
        self.vel[self.tout < 0] = 0
        self.vel[self.tout > movedur] = 0

        self.pos = movement['Amplitude'] * np.sin(2 * np.pi * movement['Frequency'] * self.tout)
        self.pos[self.tout < 0] = 0
        self.pos[self.tout > movement['Cycles'] / movement['Frequency']] = 0

        maxvel = self.params['Motor', 'Maximum speed']      # in RPM
        maxvel *= 360.0 / 60.0          # convert to deg/s

        self.duty = self.vel / maxvel / 2 + 0.5
        self.freq = np.ones_like(self.duty) * self.params['Motor', 'Pulse frequency']

        self.inhibit = np.zeros_like(self.tout, dtype=np.bool)
        self.inhibit[-10:] = True

        self.enable = np.ones_like(self.tout, dtype=np.bool)
        self.enable[self.tout < -0.5] = False
        self.enable[self.tout > movedur+0.5] = False
        self.enable[-2:] = False

        self.led = np.zeros_like(self.tout, dtype=np.bool)
        nled = int(np.round(self.params['DAQ', 'Output', 'LED pulse duration'] *
                            self.params['Motor', 'Pulse frequency']))
        for i in range(int(movement['Cycles'])+1):
            k = first(self.tout >= float(i) / movement['Frequency'])
            if k:
                self.led[k:k+nled] = True


    def run(self):
        self.digital_out_data = np.zeros_like(self.tout, dtype=np.uint32)
        self.digital_out_data = set_bit(self.digital_out_data, self.params['DAQ', 'Output', 'Inhibit line'],
                                        self.inhibit)
        self.digital_out_data = set_bit(self.digital_out_data, self.params['DAQ', 'Output', 'Enable line'],
                                        self.enable)
        self.digital_out_data = set_bit(self.digital_out_data, self.params['DAQ', 'Output', 'LED line'],
                                        self.led)

        if self.params['DAQ', 'Reference trigger'].lower() == 'none':
            trig = None
            pretrigdur = 0
        else:
            trig = self.params['DAQ', 'Reference trigger'].lower()
            pretrigdur = self.params['DAQ', 'Pretrigger duration']

        with Task() as counter_out, Task() as digital_out, Task() as analog_in, Task() as digital_in:
            # digital output
            digital_out.do_channels.add_do_chan(self.params['DAQ', 'Output', 'Digital port'],
                                                line_grouping=daq.LineGrouping.CHAN_FOR_ALL_LINES)
            digital_out.timing.cfg_samp_clk_timing(self.params['Motor', 'Pulse frequency'],
                                                   sample_mode=daq.AcquisitionType.FINITE,
                                                   samps_per_chan=len(self.digital_out_data))
            if trig is None:
                digital_out.triggers.start_trigger.cfg_dig_edge_start_trig('ai/StartTrigger',
                                                                           trigger_edge=daq.Edge.RISING)
            else:
                digital_out.triggers.start_trigger.cfg_dig_edge_start_trig(trig,
                                                                       trigger_edge=daq.Edge.RISING)
            # order is inhibit then enable
            # digital_out.write([False, True], auto_start=True)
            digital_writer = DigitalSingleChannelWriter(digital_out.out_stream)
            digital_writer.write_many_sample_port_uint32(self.digital_out_data)

            totaldur = self.duration + pretrigdur
            # analog input
            n_in_samples = int(totaldur * self.params['DAQ', 'Input', 'Sampling frequency'])
            n_in_pre_samples = int(pretrigdur * self.params['DAQ', 'Input', 'Sampling frequency'])

            aichans = [self.params['DAQ','Input', c] for c in ['SG0', 'SG1', 'SG2', 'SG3', 'SG4', 'SG5']]
            for aichan1 in aichans:
                analog_in.ai_channels.add_ai_voltage_chan(aichan1)
            analog_in.timing.cfg_samp_clk_timing(self.params['DAQ', 'Input', 'Sampling frequency'],
                                                 sample_mode=daq.AcquisitionType.FINITE,
                                                 samps_per_chan=n_in_samples)
            if trig is not None:
                analog_in.triggers.reference_trigger.cfg_dig_edge_ref_trig(self.params['DAQ', 'Reference trigger'],
                                                                           n_in_pre_samples,
                                                                           trigger_edge=daq.Edge.RISING)
            # analog_in.triggers.start_trigger.cfg_dig_edge_start_trig(self.params['DAQ', 'Start trigger'],
            #                                                          trigger_edge=daq.Edge.RISING)

            reader = AnalogMultiChannelReader(analog_in.in_stream)
            self.aidata = np.zeros((6, n_in_samples), dtype=np.float64)

            # digital input
            n_in_dig_samples = int(totaldur * self.params['DAQ', 'Input', 'Digital sampling frequency'])
            n_in_dig_pre_samples = int(pretrigdur * self.params['DAQ', 'Input', 'Digital sampling frequency'])

            digital_in.di_channels.add_di_chan(self.params['DAQ', 'Input', 'Digital input port'], '',
                                               line_grouping=daq.LineGrouping.CHAN_FOR_ALL_LINES)
            digital_in.timing.cfg_samp_clk_timing(self.params['DAQ', 'Input', 'Digital sampling frequency'],
                                                  sample_mode=daq.AcquisitionType.FINITE,
                                                  samps_per_chan=n_in_dig_samples)
            if trig is not None:
                digital_in.triggers.reference_trigger.cfg_dig_edge_ref_trig(self.params['DAQ', 'Reference trigger'],
                                                                            n_in_dig_pre_samples,
                                                                            trigger_edge=daq.Edge.RISING)
            else:
                digital_in.triggers.start_trigger.cfg_dig_edge_start_trig('ai/StartTrigger',
                                                                          trigger_edge=daq.Edge.RISING)
            # digital_in.triggers.start_trigger.cfg_dig_edge_start_trig(self.params['DAQ', 'Start trigger'],
            #                                                           trigger_edge=daq.Edge.RISING)
            digital_reader = DigitalSingleChannelReader(digital_in.in_stream)
            self.didata = np.zeros(n_in_dig_samples, dtype=np.uint32)

            # counter output
            counter_out.co_channels.add_co_pulse_chan_freq(self.params['DAQ', 'Output', 'Counter name'],
                                                           units=daq.FrequencyUnits.HZ,
                                                           idle_state=daq.Level.LOW, initial_delay=0.0,
                                                           freq=self.params['Motor', 'Pulse frequency'],
                                                           duty_cycle=0.5)
            counter_out.timing.cfg_implicit_timing(sample_mode=daq.AcquisitionType.FINITE,
                                                   samps_per_chan=len(self.duty))
            if trig is not None:
                counter_out.triggers.start_trigger.cfg_dig_edge_start_trig(trig,
                                                                           trigger_edge=daq.Edge.RISING)
            else:
                counter_out.triggers.start_trigger.cfg_dig_edge_start_trig('ai/StartTrigger',
                                                                           trigger_edge=daq.Edge.RISING)

            counter_writer = CounterWriter(counter_out.out_stream)

            counter_writer.write_many_sample_pulse_frequency(self.freq, self.duty)

            try:
                digital_out.start()
                counter_out.start()
                digital_in.start()

                analog_in.start()

                analog_in.wait_until_done(60)
                self.endTime = datetime.now()

                reader.read_many_sample(self.aidata)
                digital_reader.read_many_sample_port_uint32(self.didata)
            finally:
                pass
                # digital_out.write([True, False], auto_start=True)

            self.tin = np.arange(0, n_in_samples) / self.params['DAQ', 'Input', 'Sampling frequency']
            self.tin -= self.params['Movement', 'Wait before and after'] + pretrigdur

            self.tdig = np.arange(0, n_in_dig_samples) / self.params['DAQ', 'Input', 'Digital sampling frequency']
            self.tdig -= self.params['Movement', 'Wait before and after'] + pretrigdur

            self.forces = np.dot(self.aidata.T, self.calibration).T
            self.pwm = np.bitwise_and(self.didata, 2**self.params['DAQ', 'Input', 'PWM return line']) > 0
            self.V3Vpulse = np.bitwise_and(self.didata, 2**self.params['DAQ', 'Input', 'V3V pulse line']) > 0
            self.V3Vpulse2 = np.bitwise_and(self.didata, 2**self.params['DAQ', 'Input', 'V3V pulse2']) > 0
            self.V3Vpulse3 = np.bitwise_and(self.didata, 2**self.params['DAQ', 'Input', 'V3V pulse3']) > 0

    def incrementFileNum(self, filename):
        m = re.search('(\d+)\.h5', filename)
        if m is None:
            basename, ext = os.path.splitext(filename)
            num = 0
        else:
            basename = filename[:m.start(1)]
            num = int(m.group(1))
            ext = filename[m.end(1):]

        done = False
        while not done:
            num += 1
            filename = '{}{:03d}{}'.format(basename, num, ext)
            done = not os.path.exists(filename)

        self.filename = filename

        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)
        settings.setValue('Parameters/Output file', self.filename)

    def save(self):
        self.filename = self.params['Output file']

        if os.path.exists(self.filename):
            self.incrementFileNum(self.filename)

        with h5py.File(self.filename, 'w') as F:
            F.attrs['EndTime'] = self.endTime.strftime('%Y-%m-%d %H:%M:%S %Z')

            # save the input data
            gin = F.create_group('RawInput')
            gin.attrs['SampleFrequency'] = self.params['DAQ', 'Input', 'Sampling frequency']
            for i, aichan in enumerate(['SG0', 'SG1', 'SG2', 'SG3', 'SG4', 'SG5']):
                dset = gin.create_dataset(aichan, data=self.aidata[i, :])
                dset.attrs['HardwareChannel'] = self.params['DAQ', 'Input', aichan]

            gin = F.create_group('Calibrated')
            gin.attrs['SampleFrequency'] = self.params['DAQ', 'Input', 'Sampling frequency']
            for i, forcechan in enumerate(['xForce', 'yForce', 'zForce', 'xTorque', 'yTorque', 'zTorque']):
                dset = gin.create_dataset(forcechan, data=self.forces[i, :])
            gin.create_dataset('CalibrationMatrix', data=self.calibration)

            gin = F.create_group('DigitalInput')
            gin.attrs['SampleFrequency'] = self.params['DAQ', 'Input', 'Digital sampling frequency']
            dset = gin.create_dataset('PWMreturn', data=self.pwm)
            dset.attrs['HardwareChannel'] = self.params['DAQ', 'Input', 'Digital input port']
            dset.attrs['Line'] = self.params['DAQ', 'Input', 'PWM return line']
            dset = gin.create_dataset('V3Vpulse', data=self.V3Vpulse)
            dset.attrs['HardwareChannel'] = self.params['DAQ', 'Input', 'Digital input port']
            dset.attrs['Line'] = self.params['DAQ', 'Input', 'V3V pulse line']
            dset = gin.create_dataset('V3Vpulse2', data=self.V3Vpulse2)
            dset.attrs['HardwareChannel'] = self.params['DAQ', 'Input', 'Digital input port']
            dset.attrs['Line'] = self.params['DAQ', 'Input', 'V3V pulse2']
            dset = gin.create_dataset('V3Vpulse3', data=self.V3Vpulse3)
            dset.attrs['HardwareChannel'] = self.params['DAQ', 'Input', 'Digital input port']
            dset.attrs['Line'] = self.params['DAQ', 'Input', 'V3V pulse3']

            # save the parameters for generating the stimulus
            gout = F.create_group('NominalStimulus')
            gout.create_dataset('t', data=self.tout)
            gout.create_dataset('Position', data=self.pos)
            gout.create_dataset('Velocity', data=self.vel)

            movement = self.params.child('Movement')
            gout.attrs['Amplitude'] = movement['Amplitude']
            gout.attrs['Frequency'] = movement['Frequency']
            gout.attrs['Cycles'] = movement['Cycles']
            gout.attrs['WaitPrePost'] = movement['Wait before and after']

            # save the whole parameter tree, in case I change something and forget to add it above
            gparams = F.create_group('ParameterTree')
            self._writeParameters(gparams, self.params)

    def _writeParameters(self, group, params):
        for ch in params:
            if ch.hasChildren():
                sub = group.create_group(ch.name())
                self._writeParameters(sub, ch)
            elif ch.type() in ['float', 'int']:
                try:
                    group.attrs.create(ch.name(), ch.value())
                except TypeError as err:
                    errstr = "Error saving {} = {}: {}".format(ch.name(), ch.value(), err)
                    logging.warning(errstr)
                    self.errors.append(errstr)
                    continue
            elif ch.type() in ['list', 'str']:
                try:
                    group.attrs.create(ch.name(), str(ch.value()))
                except TypeError as err:
                    errstr = "Error saving {} = {}: {}".format(ch.name(), ch.value(), err)
                    logging.warning(errstr)
                    self.errors.append(errstr)
                    continue

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def main():
    logging.basicConfig(level=logging.DEBUG)

    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    dlg = TwiddleSetupDialog(parameters)
    if dlg.exec_() == QtWidgets.QDialog.Accepted:
        twiddle = Twiddle(dlg.params, dlg.calibration)
        twiddle.generate_sine()
        twiddle.run()
        twiddle.save()

        Wn = (twiddle.params['DAQ', 'Input', 'Smoothing cut off frequency'] /
             (0.5 * twiddle.params['DAQ', 'Input', 'Sampling frequency']))

        sos = signal.butter(5, Wn, output='sos')
        forcessm = signal.sosfiltfilt(sos, twiddle.forces, axis=-1)

        fig, ax = plt.subplots(7,1, sharex=True)
        ax[0].plot(twiddle.tout, twiddle.pos)

        for ax1, f1, lab1 in zip(ax[1:], forcessm, ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']):
            ax1.plot(twiddle.tin, f1, label=lab1)
            ax1.set_ylabel(lab1)

        # fy = twiddle.forces[1, :]
        # wind = 100
        #
        # fystd = rolling_window(fy, wind)
        # fystd = np.std(fystd, axis=-1)
        # ax[3].plot(twiddle.tin[wind-1:], fystd)

        if twiddle.params['Debug timing']:
            ispulse = np.insert(np.logical_and(twiddle.V3Vpulse[1:],
                                np.logical_not(twiddle.V3Vpulse[:-1])), 0, False)
            tpulse = twiddle.tdig[ispulse]
            if len(tpulse) >= 2:
                dtpulse = np.diff(tpulse)
                tpulse = tpulse[:-1]
            else:
                dtpulse = np.array([])

            ispulse = np.insert(np.logical_and(twiddle.V3Vpulse2[1:],
                                np.logical_not(twiddle.V3Vpulse2[:-1])), 0, False)
            tpulse2 = twiddle.tdig[ispulse]
            if len(tpulse2) >= 2:
                dtpulse2 = np.diff(tpulse2)
                tpulse2 = tpulse2[:-1]
            else:
                dtpulse2 = np.array([])

            if len(dtpulse) > 0:
                fig, ax = plt.subplots(3,1, sharex=True)
                ax[0].plot(twiddle.tout, twiddle.pos, label='motor')
                yl = ax[0].get_ylim()
                for fr, t1 in enumerate(tpulse):
                    ax[0].axvline(x=t1)
                    ax[0].annotate(str(fr+1), xy=(t1, yl[1]), horizontalalignment='center', verticalalignment='top')
                ax[0].set_ylabel('Motor position')

                ax[1].plot(twiddle.tdig, twiddle.V3Vpulse2+3, label='Exp')
                ax[1].plot(twiddle.tdig, twiddle.V3Vpulse3+2, label='E')
                ax[1].plot(twiddle.tdig, twiddle.V3Vpulse+1, label='F')
                ax[1].legend()
                ax[1].set_ylabel('Pulses')

                ax[2].plot(tpulse, dtpulse, 'o', label='F')
                if len(dtpulse2) != 0:
                    ax[2].plot(tpulse2, dtpulse2, '+', label='exposure')

                ax[2].set_ylabel('Pulse period (s)')
                ax[2].set_xlabel('Time (s)')

                logging.debug('{} pulses on line 1. {} pulses on line 2'.format(len(tpulse), len(tpulse2)))
                weird = [str(k) for k in np.argwhere(abs((dtpulse - dtpulse[-1])/dtpulse[-1]) > 0.05) + 2]
                logging.debug('Weird pulses {}'.format(','.join(weird)))

        plt.show()

        vidfilestr = os.path.join(twiddle.params['Video file directory'], twiddle.params['Video file base name'])
        vidfiles = glob(vidfilestr)
        vidtimes = [os.path.getmtime(vidfile) for vidfile in vidfiles]

        endTime = time.mktime(twiddle.endTime.timetuple())
        islater = [vidtime > endTime for vidtime in vidtimes]

        if not any(islater):
            QtWidgets.QMessageBox.information(None, 'Video file', 'Remember to stop the video camera recording',
                                              QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

            vidfiles = glob(vidfilestr)
            vidtimes = [os.path.getmtime(vidfile) for vidfile in vidfiles]

            islater = [vidtime > endTime for vidtime in vidtimes]

        if not any(islater):
            QtWidgets.QMessageBox.information(None, 'Video file', 'Could not find current video file')
        else:
            ind = np.argmax(np.array(vidtimes) * np.array(islater).astype(np.float))
            vidfile = vidfiles[ind]
            _, vidfileext = os.path.splitext(vidfile)

            _, outfilename = os.path.split(twiddle.filename)
            outfilename, _ = os.path.splitext(outfilename)
            newvidfilename = os.path.join(twiddle.params['Video file directory'], outfilename + vidfileext)

            os.rename(vidfile, newvidfilename)


   # return app.exec_()


if __name__ == '__main__':
    sys.exit(main())

