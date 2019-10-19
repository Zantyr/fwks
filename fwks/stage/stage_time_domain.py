"""
Operations in time domain
"""

import keras
import numpy as np
import sklearn.linear_model as lm
import pyAudioAnalysis.audioFeatureExtraction as aF

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

    def output_dtype(self, input_dtype):
        skip = 4
        return DType("Array", [input_dtype.shape[0] // self.hop - 4, 14], np.float32)

    def _function(self, recording):
        skip = 4
        time_frames = len(recording) // self.hop - skip
        features = np.zeros([time_frames, 14], np.float32)
        for time in range(time_frames):
            frame = recording[self.hop * time : self.hop * time + self.fft_size]
            X_new = np.abs(np.fft.rfft(frame))
            X_prev = X if time else np.zeros_like(np.fft.rfft)
            X = X_new
            features[time, 0] = aF.stZCR(frame)
            features[time, 1] = aF.stEnergy(frame)
            features[time, 2] = aF.stEnergyEntropy(frame)
            features[time, 3:5] = aF.stSpectralCentroidAndSpread(X, self.sr)
            features[time, 5] = aF.stSpectralEntropy(X)
            features[time, 6] = aF.stSpectralRollOff(X, 0.85, self.sr)
            features[time, 7] = aF.stSpectralFlux(X, X_prev)
        features[:, 8:14] = self.pseudoformants(recording)
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
    """

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


class WindowedTimeFrequencyFBank():
    def __init__(self, preset="stft", summary="max"):
        pass
