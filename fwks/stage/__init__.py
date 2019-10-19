import abc
import gammatone.gtgram
import hashlib
import keras
import keras.backend as K
import librosa
import numpy as np
import scipy as sp
import scipy.signal as sps

from keras.layers import Input, Lambda
from keras.models import Model
from syntax import Show

from .stage_meta import SelectionAdapter, Stage, NetworkableMixin, Loss, Analytic, Neural, ToDo, DType, CustomNeural, Normalizer
from .stage_selection_adapter import RandomSelectionAdapter, SpeakerSelectionAdapter
from .stage_loss import CTCLoss, L2Loss
from .stage_transforms import PlainPowerFourier, LogPowerFourier, TrainableCQT, TrainableCZT, CZT, CQT, CommonFateTransform, DCT, Cochleagram
from .stage_preprocessing import Window, EqualLoudnessWeighting, PCENScaling, AdaptiveGainAndCompressor, OverlapAdd
from .stage_filterbanks import TriangularERB, HarmonicTriangularERB, OverlappingHarmonicTriangularERB, RoEx, GammatoneFilterbank, MelFilterbank
from .stage_time_domain import (GammaChirp, TimeRoex, TrainableConvolve, CARFAC,
    PLC, HandCrafted, WindowedTimeFrequencyFBank
)
from .stage_neural import (EarlyDNN, EarlyConv2D, EarlyConv1D, SparseDNN,
    AntimonotonyLayer, RNN, LaterConv1D, LaterDNN, LaterSparse1D, TimeWarpingRNN,
    TimeWarpingCNN, Core, CNN2D, LearnableFilterbank
)
from .stage_misc import LogPower, ConcatFeatures
from .stage_normalizers import MeanStdNormalizer


class AbstractWavelet(ToDo):
    def __init__(self):
        pass


class AbstractFilter(ToDo):
    """
    Will represent phase shifts in the cochlea
    """


class ExcitationTrace(ToDo):
    """
    Add max of current and exponential smoothing of past features at each band
    """


def phonemic_map(phones, activation='softmax'):
    inp = keras.layers.Input((None, 512))
    outp = keras.layers.Dense(phones + 1, activation=activation)(inp)
    return CustomNeural(keras.models.Model(inp, outp))
