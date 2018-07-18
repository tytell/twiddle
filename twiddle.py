import sys, os
import logging
import re
from datetime import datetime
from time import strftime

import nidaqmx.constants as daq
from nidaqmx.stream_writers import CounterWriter
from nidaqmx.stream_readers import AnalogMultiChannelReader, DigitalSingleChannelReader
from nidaqmx import Task

from PyQt5 import QtWidgets, QtCore

import numpy as np
import matplotlib.pyplot as plt

import h5py

from twiddleSetup import TwiddleSetupDialog, parameters, SETTINGS_FILE

class Twiddle(object):
    def __init__(self, params, calibration):
        self.params = params
        self.calibration = calibration

        self.errors = []

    def generate_sine(self):
        movement = self.params.child('Movement')

        self.duration = 2*movement['Wait before and after'] + \
                        movement['Cycles']/movement['Frequency']

        dt = 1.0/self.params['Motor', 'Pulse frequency']
        self.tout = np.arange(0, self.duration, dt)
        self.tout -= movement['Wait before and after']

        self.vel = 2*np.pi * movement['Frequency'] * movement['Amplitude'] * \
            np.cos(2 * np.pi * movement['Frequency'] * self.tout)
        self.vel[self.tout < 0] = 0
        self.vel[self.tout > movement['Cycles']/movement['Frequency']] = 0

        self.pos = movement['Amplitude'] * np.sin(2 * np.pi * movement['Frequency'] * self.tout)
        self.pos[self.tout < 0] = 0
        self.pos[self.tout > movement['Cycles'] / movement['Frequency']] = 0

        maxvel = self.params['Motor', 'Maximum speed']      # in RPM
        maxvel *= 360.0 / 60.0          # convert to deg/s

        self.duty = self.vel / maxvel / 2 + 0.5
        self.freq = np.ones_like(self.duty) * self.params['Motor', 'Pulse frequency']

    def run(self):
        with Task() as counter_out, Task() as digital_out, Task() as analog_in, Task() as digital_in:
            # digital output
            doname = '{0}/line{1},{0}/line{2}'.format(self.params['DAQ', 'Output', 'Digital port'],
                                                      self.params['DAQ', 'Output', 'Inhibit line'],
                                                      self.params['DAQ', 'Output', 'Enable line'])

            digital_out.do_channels.add_do_chan(doname,
                                                line_grouping=daq.LineGrouping.CHAN_PER_LINE)
            # order is inhibit then enable
            digital_out.write([False, True], auto_start=True)

            # analog input
            n_in_samples = int(self.duration * self.params['DAQ', 'Input', 'Sampling frequency'])

            aichans = [self.params['DAQ','Input', c] for c in ['SG0', 'SG1', 'SG2', 'SG3', 'SG4', 'SG5']]
            for aichan1 in aichans:
                analog_in.ai_channels.add_ai_voltage_chan(aichan1)
            analog_in.timing.cfg_samp_clk_timing(self.params['DAQ', 'Input', 'Sampling frequency'],
                                                 sample_mode=daq.AcquisitionType.FINITE,
                                                 samps_per_chan=n_in_samples)
            reader = AnalogMultiChannelReader(analog_in.in_stream)
            self.aidata = np.zeros((6, n_in_samples), dtype=np.float64)

            # digital input
            n_in_dig_samples = int(self.duration * self.params['DAQ', 'Input', 'Digital sampling frequency'])

            digital_in.di_channels.add_di_chan(self.params['DAQ', 'Input', 'Digital input port'], '',
                                               line_grouping=daq.LineGrouping.CHAN_FOR_ALL_LINES)
            digital_in.timing.cfg_samp_clk_timing(self.params['DAQ', 'Input', 'Digital sampling frequency'],
                                                  sample_mode=daq.AcquisitionType.FINITE,
                                                  samps_per_chan=n_in_dig_samples)
            digital_in.triggers.start_trigger.cfg_dig_edge_start_trig("ai/StartTrigger",
                                                                      trigger_edge=daq.Edge.RISING)
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
            counter_out.triggers.start_trigger.cfg_dig_edge_start_trig('ai/StartTrigger',
                                                                       trigger_edge=daq.Edge.RISING)

            counter_writer = CounterWriter(counter_out.out_stream)

            counter_writer.write_many_sample_pulse_frequency(self.freq, self.duty)

            try:
                counter_out.start()
                digital_in.start()
                self.startTime = datetime.now()

                analog_in.start()

                analog_in.wait_until_done(10)
                reader.read_many_sample(self.aidata)
                digital_reader.read_many_sample_port_uint32(self.didata)
            finally:
                digital_out.write([True, False], auto_start=True)

            self.tin = np.arange(0, n_in_samples) / self.params['DAQ', 'Input', 'Sampling frequency']
            self.tin -= self.params['Movement', 'Wait before and after']

            self.tdig = np.arange(0, n_in_dig_samples) / self.params['DAQ', 'Input', 'Digital sampling frequency']
            self.tdig -= self.params['Movement', 'Wait before and after']

            self.forces = np.dot(self.aidata.T, self.calibration).T
            self.pwm = np.bitwise_and(self.didata, 2**self.params['DAQ', 'Input', 'PWM return line']) > 0
            self.V3Vpulse = np.bitwise_and(self.didata, 2**self.params['DAQ', 'Input', 'V3V pulse line']) > 0

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
            F.attrs['StartTime'] = self.startTime.strftime('%Y-%m-%d %H:%M:%S %Z')

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

        fig, ax = plt.subplots(3,1, sharex=True)
        ax[0].plot(twiddle.tout, twiddle.pos)
        ax[1].plot(twiddle.tin, twiddle.forces[5, :])
        ax[2].plot(twiddle.tin, twiddle.forces[0, :])

        plt.show()

   # return app.exec_()


if __name__ == '__main__':
    sys.exit(main())

