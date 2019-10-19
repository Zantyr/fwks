import numpy as np
import scipy as sp
import scipy.signal as sps

from .stage_meta import Analytic, DType, ToDo


class Window(Analytic):
    """
    Divide recording into uniform overlapping windows
    Those can form a basis to different transforms
    Can apply windowing function
    """
    def __init__(self, size, hop, win_func=None):
        self.size = size
        self.hop = hop
        self.win_func = win_func  # if not None, initialize
        self.previous = None

    def output_dtype(self, input_dtype):
        return DType("Array", [1 + input_dtype.shape[0] // self.hop, self.size], np.float32)

    def _function(self, recording):
        windowed = np.zeros([1 + recording.shape[0] // self.hop, self.size], np.float32)
        for ix in range(windowed.shape[0]):
            slice = recording[ix * self.hop : ix * self.hop + self.size]
            if len(slice) != self.size:
                slice = np.pad(slice, (0, self.size - len(slice)), 'constant')
            if self.win_func is not None:
                slice = self.win_func * slice
            windowed[ix, :] = slice
        return windowed


class OverlapAdd(Analytic):
    """
    Reverse of windowing operation
    """
    def __init__(self, size, hop, win_func=None):
        self.size = size
        self.hop = hop
        self.win_func = win_func  # if not None, initialize
        self.previous = None

    def output_dtype(self, input_dtype):
        return DType("Array", [recording.shape[0] * self.hop + self.size], np.float32)

    def _function(self, recording):
        windowed = np.zeros([recording.shape[0] * self.hop + self.size], np.float32)
        if self.win_func:
            window = self.win_func(recording.shape[1])
        else:
            window = 1
        for ix in range(windowed.shape[0]):
            windowed[ix * self.hop : ix * self.hop + self.size] = window * recording[ix]
        return windowed


class EqualLoudnessWeighting(Analytic):
    """
    IEC 61672:2003
    Based on: https://gist.github.com/endolith/148112
    """
    def __init__(self, kind):
        assert kind in ["A", "B", "C"]
        if kind == "A":
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 0.17
            numerator = [(2*np.pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
            denominator = sp.polymul([1, 4*np.pi * f4, (2*np.pi * f4)**2],
                           [1, 4*np.pi * f1, (2*np.pi * f1)**2])
            denominator = sp.polymul(sp.polymul(denominator, [1, 2*np.pi * f3]), [1, 2*np.pi * f2])
        if kind == "B":
            f1 = 20.598997
            f2 = 158.5
            f4 = 12194.217
            A1000 = 1.9997
            numerator = [(2*np.pi * f4)**2 * (10**(A1000/20)), 0, 0, 0]
            denominator = sp.polymul([1, 4*np.pi * f4, (2*np.pi * f4)**2],
                           [1, 4*np.pi * f1, (2*np.pi * f1)**2])
            denominator = sp.polymul(denominator, [1, 2*np.pi * f2])
        if kind == "C":
            f1 = 20.598997 
            f4 = 12194.217
            C1000 = 0.0619
            numerator = [(2*np.pi*f4)**2*(10**(C1000/20.0)),0,0]
            denominator = sp.polymul([1,4*np.pi*f4,(2*np.pi*f4)**2.0],[1,4*np.pi*f1,(2*np.pi*f1)**2])
        self.filter = sps.bilinear(numerator, denominator, 16000)

    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return input_dtype
        
    def _function(self, recording):
        return sps.filtfilt(*self.filter, recording)


class PCENScaling(Analytic):
    """
    Per Channel Energy Normalization
    http://www.justinsalamon.com/uploads/4/3/9/4/4394963/lostanlen_pcen_spl2018.pdf
    Also based on librosa for parameters
    
    sps.lfilter with [b] [1, b-1] is used to smooth, it is generally an exponential decay filter
    """
    def __init__(self, sr=16000, hop=128, time_constant=0.4):
        self.alpha = 0.98
        self.delta = 2
        self.r = 0.5
        self.epsilon = 1e-6
        t_frames = time_constant * sr / hop
        self.b = b = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)
    
    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return input_dtype
    
    def _function(self, spec):
        spec_filtered = sps.lfilter([self.b], [1, self.b - 1], spec)
        return ((spec / (self.epsilon + spec_filtered) ** self.alpha ) + self.delta) ** self.r - self.delta ** self.r
        

class AdaptiveGainAndCompressor(Analytic):
    """
    In spectrogram domain
    """

    def __init__(self, sr=16000, hop=128, compression_factor=0.2, time_constant=0.4):
        t_frames = time_constant * sr / hop
        self.epsilon = 1e-6
        self.b = b = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)
        self.compression_factor = compression_factor
        
    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return input_dtype
    
    def _function(self, spec):
        # first, blur in time to get the mean volume level
        # then divide by this
        # the compress by non-linear maximum, the lower the compfactor, the more compressive the compressor is
        spec_filtered = np.abs(sps.lfilter([self.b], [1, self.b - 1], spec))
        return (np.abs(spec) / (self.epsilon + spec_filtered)) ** self.compression_factor
