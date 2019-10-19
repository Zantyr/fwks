"""
Everything that summarizes Fourier (or other transforms)
into another frequency scale and summarizes frequency ranges
"""

import gammatone
import librosa
import numpy as np

from .stage_meta import Analytic, DType, ToDo


class FreqFilter(Analytic):
    def __init__(self, n_filts=24, sr=16000, window=512, hop=128, sum="log-mean"):
        fbank = lambda freq, bounds: self._fbank(freq, bounds, sr, window)
        self.scale = self._scale(sr, n_filts, 20)
        self.scale = np.array(sorted(self.scale))
        self.widths = np.concatenate([[0], self.scale, [sr]])
        self.widths = [(self.widths[x], self.widths[x + 2]) for x in range(len(self.scale))]
        self.filter = np.array([fbank(self.scale[x], self.widths[x]) for x in range(n_filts)], np.float32).T
        self.sr = sr
        self.n_filts = n_filts

    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return DType("Array", [input_dtype.shape[0], self.n_filts], np.float32)
    
    def _function(self, recording):
        return np.dot(recording, self.filter)
    

class ERBFilter(FreqFilter):
    """
    ERBFilter base class, filter function should be supplied.
    """
    _scale = lambda self, *args: gammatone.gtgram.centre_freqs(*args)

    
class TriangularERB(ERBFilter):
    def _fbank(_, freq, bounds, sr, window):
        def filt_fn(x):
            if x < bounds[0] or x > bounds[1]:
                return 0.
            if x >= freq:
                return 1. - (x - freq) / (bounds[1] - freq)
            if x < freq:
                return 1. - (freq - x) / (freq - bounds[0])
        filt_fn = np.vectorize(filt_fn)
        num_window = window // 2 + 1
        filt = filt_fn(np.arange(num_window) * sr / window)
        return filt / filt.sum()

    
class HarmonicTriangularERB(ERBFilter):
    """
    Apply the triangular filter and then scale in time (with 2n 3n 4n... speed)
    Then add it with weights
    Then scale to unit max
    """
    def _fbank(_, freq, bounds, sr, window):
        def filt_fn(x):
            if x < bounds[0] or x > bounds[1]:
                return 0.
            if x >= freq:
                return 1. - (x - freq) / (bounds[1] - freq)
            if x < freq:
                return 1. - (freq - x) / (freq - bounds[0])
        filt_fn = np.vectorize(filt_fn)
        num_window = window // 2 + 1
        filt = np.zeros(num_window, np.float32)
        for i in range(5):
            filt += filt_fn(np.arange(num_window) * sr / window / (i + 1)) / (i+1)
        return filt / filt.sum()


class OverlappingHarmonicTriangularERB(ERBFilter):
    """
    As HarmonicTraingularERB but with other bounds on the filters
    """
    def _fbank(_, freq, bounds, sr, window):
        bounds = freq - bounds[0], freq + bounds[1]
        def filt_fn(x):
            if x < bounds[0] or x > bounds[1]:
                return 0.
            if x >= freq:
                return 1. - (x - freq) / (bounds[1] - freq)
            if x < freq:
                return 1. - (freq - x) / (freq - bounds[0])
        filt_fn = np.vectorize(filt_fn)
        num_window = window // 2 + 1
        filt = np.zeros(num_window, np.float32)
        for i in range(5):
            filt += filt_fn(np.arange(num_window) * sr / window / (i + 1)) / (i+1)
        return filt / filt.sum()

    
class MelFilterbank(Analytic):
    def __init__(self, n_mels, fmin=20, fmax=8000, sr=16000):
        self.mel_basis = None
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.sr = sr
    
    def output_dtype(self, input_dtype):
        # print(input_dtype, "MELTI")
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        self.mel_basis = librosa.filters.mel(self.sr, (input_dtype.shape[1] - 1) * 2, self.n_mels, self.fmin, self.fmax).T
        return DType("Array", [input_dtype.shape[0], self.n_mels], np.float32)
    
    def _function(self, recording):
        return np.dot(recording, self.mel_basis)


class TuningCurves(ToDo):
    """
    How to implement that?
    """
    
    

class RoEx(ToDo):
    """
    This is general filter but where weights are an integrated bell function
    """    

    def _fbank(_, freq, bounds, sr, window):
        pass

    
    
class GammatoneFilterbank(Analytic):
    
    _sums = {
        "mean": lambda x: np.mean(x, axis=0),
        "log-mean": lambda x: np.log(np.abs(np.mean(x, axis=0))),
        "log-max": lambda x: np.log(np.max(np.abs(x), axis=0)),
    }
    
    def __init__(self, n_mels=24, sr=16000, window=512, hop=128, sum="log-mean"):
        self.scale = gammatone.gtgram.centre_freqs(sr, n_mels, 20)
        self.filterbank = gammatone.gtgram.make_erb_filters(sr, self.scale)
        self.window = window
        self.hop = hop
        self.sum = self._sums[sum]
        self.n_mels = n_mels
    
    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return DType("Array", [int(np.ceil(1 + (input_dtype.shape[0] - self.window) / self.hop)), self.n_mels], np.float32)
        
    def _function(self, recording):
        gt = gammatone.gtgram.erb_filterbank(recording[:], self.filterbank).T
        length = int(np.ceil(1 + (gt.shape[0] - self.window) / self.hop))
        ret = np.zeros([length, gt.shape[1]], np.float32)
        for i in range(length):
            ret[i, :] = self.sum(gt[i * self.hop : i * self.hop + self.window, :])
        return ret


class CARFAC(ToDo):
    pass

    