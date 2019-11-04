"""
Operations in time domain
"""

import keras
import numpy as np
import sklearn.linear_model as lm
import pyAudioAnalysis.audioFeatureExtraction as aF
import scipy as sp
import scipy.signal

from .stage_meta import ToDo, Neural, Analytic, DType


class GammaChirp(ToDo):
    def __init__(self):
        pass


class TimeRoex(ToDo):
    pass


class TrainableConvolve(Neural):
    def __init__(self, n_channels=48, winsize=512, stride=128, wavelet_constraint=True):
        self.n_channels = n_channels
        self.winsize = winsize
        self.stride = stride
        self.wavelet_constraint = wavelet_constraint

    def new_network(self, dtype):
        first = inp = keras.layers.Input(shape=[None] + self.n_channels)
        if len(dtype.shape) > 2:
            first = keras.layers.Flatten()(first)
        if self.wavelet_constraint:
            constraint = lambda x: (x - K.mean(x)) / K.std(x)
        else:
            constraint = None
        outp = keras.layers.Conv1D(self.n_channels, self.winsize, strides=self.stride, kernel_constraint=constraint, bias=(not self.wavelet_constraint))
        mdl = Model(first, outp)
        return mdl


class TrainableWavelet(ToDo):
    """
    This would require an implementation of custom layer
    """


class CARFAC(ToDo):
    pass


class PLC(Analytic):
    def __init__(self, num_banks=24):
        self.num_banks = num_banks

    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return DType("Array", [input_dtype.shape[0], self.num_banks], np.float32)

    def _function(self, recording):
        shape = list(recording.shape)
        shape[1] = self.num_banks
        mapped = np.zeros(shape, np.float32)
        window_len = recording.shape[1] - self.num_banks
        for i in range(shape[0]):
            frame = recording[i]
            windows = np.zeros([window_len, self.num_banks], np.float32)
            answers = np.zeros([window_len], np.float32)
            for time in range(window_len):
                windows[time] = frame[time : time + self.num_banks]
                answers[time] = frame[time + self.num_banks]
            model = lm.LinearRegression(False)
            model.fit(windows, answers)
            mapped[i] = model.coef_
        return mapped


class HandCrafted(Analytic):
    def __init__(self, sr=16000):
        self.sr = sr
        self.hop = 128
        self.fft_size = 512
        self.skip = 4

    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return DType("Array", [input_dtype.shape[0], 14], np.float32)

    def _function(self, recording):
        time_frames = recording.shape[0]
        features = np.zeros([time_frames, 14], np.float32)
        for time in range(time_frames):
            frame = recording[time, :]
            X_new = np.abs(np.fft.rfft(frame))
            X_prev = X if time else np.zeros_like(np.fft.rfft)
            X = X_new
            features[time, 0] = aF.stZCR(frame)
            features[time, 1] = aF.stEnergy(frame)
            features[time, 2] = aF.stEnergyEntropy(frame)
            features[time, 3:5] = aF.stSpectralCentroidAndSpread(X + 2e-12, self.sr)
            features[time, 5] = aF.stSpectralEntropy(X)
            features[time, 6] = aF.stSpectralRollOff(X, 0.85, self.sr)
            features[time, 7] = aF.stSpectralFlux(X, X_prev)
            features[time, 8:14] = HandCrafted.formants(frame) / 1000 # division for normalization (results in kHz)
        return features

    def pseudoformants(self, wave):
        skip = 4
        formmap = np.zeros([len(wave) // self.hop - skip, 6], dtype=np.float32)
        for time in range(len(wave) // self.hop - skip):
            frame = wave[time * self.hop : time * self.hop + self.fft_size]
            formmap[time, :] = HandCrafted.formants(frame)
        # averaging for smoother trajectories
        for line in range(formmap.shape[1]):
            for time in range(len(wave) // self.hop - skip - 10):
                formmap[5 + time, line] = formmap[time : time + 11, line].mean()
        return formmap

    @staticmethod
    def formants(frame, n_forms=6, interbin_fq=31.25, coeffs=24):
        forms = np.zeros(n_forms, dtype=np.float32)
        spec = np.log(np.abs(np.fft.rfft(frame)[1:]) + 2e-15)
        ceps = np.fft.rfft(spec)
        ceps[coeffs:] = 0
        spec2 = np.fft.irfft(ceps)
        candidates = np.where((spec2[:-2] < spec2[1:-1]) & (spec2[1:-1] > spec2[2:]))[0][:n_forms] * interbin_fq
        forms[:len(candidates)] = candidates
        return forms


class WindowedTimeFrequencyFBank(Analytic):
    """
    STFT -> filters....
    Accepts functions that maintain the state and return the state
    Initial state is None and is not passed to function returning single scalar
    Functions are tested for accepting the state beforehand
    """

    def __init__(self, fft_size=512, hop=128, summary="max", ripple=10, level=3):
        self.ripple = ripple
        self.level = level
        self.fft_size = fft_size
        self.hop = hop
        self.func = (lambda x: np.max(np.abs(x))) if summary == 'max' else summary
        self._stateful = False
        try:
            isinstance(self.func(np.arange(10), None), tuple)
            self._stateful = True
        except:
            pass # function is not stateful

    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return DType("Array", [
            (input_dtype.shape[0] - self.fft_size) // self.hop , self.n_channels], np.float32)

    def _function(self, recording):
        time_range = (len(recording) - self.fft_size) // self.hop 
        spec = np.zeros([time_range, self.n_channels], np.float32)
        for f in range(self.n_channels):
            if f == 0:
                b, a = sp.signal.cheby1(self.level, self.ripple, (f + 1) / self.n_channels, 'lowpass')
            elif f == self.n_channels - 1:
                b, a = sp.signal.cheby1(self.level, self.ripple, (f / self.n_channels), 'highpass')
            else:
                b, a = sp.signal.cheby1(self.level, self.ripple, [(f / self.n_channels), ((f + 1) / self.n_channels)], 'bandpass')
            channel = sp.signal.lfilter(b, a, recording)
            if self._stateful:
                state = None

                for time in range(time_range):
                    spec[time, f], state = self.func(channel[time * self.hop : time * self.hop + self.fft_size], state)
            else:
                for time in range(time_range):
                    spec[time, f] = self.func(channel[time * self.hop : time * self.hop + self.fft_size])
        return spec

    @property
    def n_channels(self):
        return self.fft_size // 2 + 1
    
