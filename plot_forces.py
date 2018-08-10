import sys
import h5py
import numpy as np
from scipy import interpolate, signal
import matplotlib.pyplot as plt

def fourierintegral(t, y, freq):
    Y = []
    for freq1 in freq:
        s = np.exp(-2*np.pi*1j*t*freq)

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
        Aest.append(np.abs(Aest1) / (t[-1] - t[0]))


class TwiddleData():
    def __init__(self, filename, ledframe=0):
        self._load(filename, ledframe)

    def _load(self, filename, ledframe=0):
        with h5py.File(filename, 'r') as h5file:
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

            self.tnorm = self.t * self.freq

            tstim = np.array(h5file['/NominalStimulus/t'])
            pos = np.array(h5file['/NominalStimulus/Position'])

            inrange = np.logical_and(self.t >= tstim[0], self.t <= tstim[-1])
            self.pos = np.zeros_like(self.t)
            self.pos[inrange] = interpolate.interp1d(tstim, pos, assume_sorted=True)(self.t[inrange])

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

    def get_forces(self):
        pass

def main():
    test_fourier()

    datafile = 'D:\\Twiddlefish\\Raw data\\18E\\baseline1HzB_004.h5'
    td = TwiddleData(datafile, 317)

    istime = np.logical_and(td.t >= 0, td.t <= td.ncycles/td.freq)
    f, Txpow = signal.periodogram(td.torque[0, istime], fs=td.sampfreq)
    f, Typow = signal.periodogram(td.torque[1, istime], fs=td.sampfreq)
    f, Tzpow = signal.periodogram(td.torque[2, istime], fs=td.sampfreq)

    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].plot(f, Txpow)
    ax[1].plot(f, Typow)
    ax[2].plot(f, Tzpow)
    ax[2].set_xlim(0, 10)

    sos = signal.butter(9, 8/(td.sampfreq/2), output='sos')

    Txs = signal.sosfiltfilt(sos, td.torque[0, :])
    Tys = signal.sosfiltfilt(sos, td.torque[1, :])
    Tzs = signal.sosfiltfilt(sos, td.torque[2, :])

    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].plot(td.t, td.torque[0, :])
    ax[0].plot(td.t, Txs)
    ax[1].plot(td.t, td.torque[1, :])
    ax[1].plot(td.t, Tys)
    ax[2].plot(td.t, td.torque[2, :])
    ax[2].plot(td.t, Tzs)

    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())
