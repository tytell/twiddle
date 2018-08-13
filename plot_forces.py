import sys
import h5py
import numpy as np
from scipy import interpolate, signal
import matplotlib.pyplot as plt
import itertools

def fourierintegral(t, y, freq):
    y -= np.mean(y)

    Y = []
    for freq1 in freq:
        s = np.exp(-2*np.pi*1j*t*freq1)

        Y.append(np.trapz(y * s, t))
    return Y

def test_fourier():
    fs = 100.0
    t = np.arange(0,5*fs, 1.0/fs)
    freq = 2.3

    A = [0.2, 0.5, 1, 2, 10]
    ph = [0, 0.3, 0.9, 0.2, 0.4]

    Aest = []
    for A1, ph1 in zip(A, ph):
        sig = A1*np.sin(2*np.pi*(freq*t + ph1))
        Aest1 = fourierintegral(t, sig, [freq])
        Aest.append(2*np.abs(Aest1) / (t[-1] - t[0]))


class TwiddleData():
    def __init__(self, filename, ledframe=0):
        self._load(filename, ledframe)

    def _load(self, filename, ledframe=0, momentarm=0.18):
        with h5py.File(filename, 'r') as h5file:
            if 'PositionAmplitude' in h5file['/NominalStimulus'].attrs:
                self.amp = h5file['/NominalStimulus'].attrs['PositionAmplitude']
            else:
                self.amp = h5file['/NominalStimulus'].attrs['Amplitude']
            self.freq = h5file['/NominalStimulus'].attrs['Frequency']
            self.ncycles = h5file['/NominalStimulus'].attrs['Cycles']
            waitbefore = h5file['/NominalStimulus'].attrs['WaitPrePost'] + \
                h5file['/ParameterTree/DAQ'].attrs['Pretrigger duration']

            fx = np.array(h5file['/Calibrated/xForce'])
            fy = np.array(h5file['/Calibrated/yForce'])
            fz = np.array(h5file['/Calibrated/zForce'])
            self.force = np.vstack((fx, fy, fz))

            tx = np.array(h5file['/Calibrated/xTorque'])
            ty = np.array(h5file['/Calibrated/yTorque'])
            tz = np.array(h5file['/Calibrated/zTorque'])
            self.torque = np.vstack((tx, ty, tz))

            fs = h5file['/Calibrated'].attrs['SampleFrequency']
            self.t = np.arange(0, len(tx)) / fs
            self.t -= waitbefore
            self.sampfreq = fs

            self.force0 = np.mean(self.force[:, self.t < 0], axis=1)
            self.force -= self.force0[:, np.newaxis]

            self.torque0 = np.mean(self.torque[:, self.t < 0], axis=1)
            self.torque -= self.torque0[:, np.newaxis]
            self.forces = None
            self.momentarm = momentarm

            self.tnorm = self.t * self.freq

            tstim = np.array(h5file['/NominalStimulus/t'])
            pos = np.array(h5file['/NominalStimulus/Position'])
            vel = np.array(h5file['/NominalStimulus/Velocity'])

            inrange = np.logical_and(self.t >= tstim[0], self.t <= tstim[-1])
            self.pos = np.zeros_like(self.t)
            self.pos[inrange] = interpolate.interp1d(tstim, pos, assume_sorted=True)(self.t[inrange])

            self.vel = np.zeros_like(self.t)
            self.vel[inrange] = interpolate.interp1d(tstim, vel, assume_sorted=True)(self.t[inrange])

            fsdig = h5file['/DigitalInput'].attrs['SampleFrequency']
            Fline = h5file['/DigitalInput/V3Vpulse']

            # look for where F goes high
            indF = np.argwhere(np.logical_and(Fline[1:], np.logical_not(Fline[:-1])))
            self.tpair = (indF[:, 0].astype(float)+1) / fsdig
            self.tpair -= waitbefore

            # find the last pulse before movement
            p0 = np.argwhere(self.tpair <= 0)[-1, 0]
            pairnum = np.arange(0, len(indF)) - p0 + ledframe

            self.pairnum = np.zeros_like(self.t)
            for t1, t2, k in zip(self.tpair, self.tpair[1:], pairnum):
                ist = np.logical_and(self.t >= t1, self.t < t2)
                self.pairnum[ist] = k

    def smooth_forces(self, cutoffmul=8):
        cutoff = self.freq * cutoffmul

        sos = signal.butter(9, cutoff / (self.sampfreq / 2), output='sos')

        self.torques = signal.sosfiltfilt(sos, self.torque, axis=1)

    def get_forces(self, cutoffmul=8, peakpercentile=100.0):
        if self.torques is None:
            self.smooth_forces(cutoffmul=cutoffmul)

        forces = self.torques / self.momentarm

        # vel is angular velocity in deg/sec
        power = self.torques[2, :] * np.deg2rad(self.vel)

        thrustpeak = []
        thrustpeaktime = []
        thrustmean = []
        thrustimp = []

        latpeak = []
        latmean = []
        latimp = []

        powerpeak = []
        powermean = []
        powertot = []
        for c, s in zip(np.arange(0, self.ncycles, 0.5), itertools.cycle([1, -1])):
            istime = np.logical_and(self.t >= c*self.freq, self.t < (c+0.5)*self.freq)
            thrustpeak.append(np.max(forces[0, istime]))

            thrustmean.append(np.mean(forces[0, istime]))
            thrustimp.append(np.trapz(forces[0, istime], self.t[istime]))

            latpeak.append(np.percentile(s * forces[1, istime], peakpercentile))
            latmean.append(np.mean(s * forces[1, istime]))
            latimp.append(np.trapz(s * forces[1, istime], self.t[istime]))

            powerpeak.append(np.percentile(power[1, istime], peakpercentile))
            powermean.append(np.mean(power[1, istime]))
            powertot.append(np.trapz(power[1, istime], self.t[istime]))


def main():
    # test_fourier()

    datafile = 'D:\\Twiddlefish\\Raw data\\18E\\TallerEqLen1_3HzB_002.h5'
    td = TwiddleData(datafile, 23)

    td.smooth_forces(cutoffmul=8)
    td.get_forces()

    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].plot(td.t, td.torque[0, :])
    ax[0].plot(td.t, td.torques[0, :])

    ax[1].plot(td.t, td.torque[1, :])
    ax[1].plot(td.t, Tys)
    ax[1].axhline(Tyamp, color='r')

    ax[2].plot(td.t, td.torque[2, :])
    ax[2].plot(td.t, Tzs)
    ax[2].axhline(Tzamp, color='r')

    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())
