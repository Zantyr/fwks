"""
All later layers of the network
"""

import keras
from keras.initializers import Orthogonal
from keras.models import Model
from tensorflow.spectral import dct

import keras.backend as K
import numpy as np

from fwks.model.meta import _defaults

from .stage_filterbanks import TriangularERB
from .stage_meta import ToDo, Neural, Analytic


class DNN(Neural):
    def __init__(self, size=512, depth=3):
        self.depth = depth
        self.size = size

    def get_graph(self):
        def maker(inp):
            if len(inp.shape) > 3:
                inp = keras.layers.TimeDistributed(keras.layers.Flatten())(inp)
            for i in range(self.depth):
                inp = keras.layers.Dense(self.size, activation='linear')(inp)
                inp = keras.layers.LeakyReLU(0.01)(inp)
                inp = keras.layers.BatchNormalization()(inp)
            return inp
        return maker


class CNN1D(Neural):
    def __init__(self, channels=256, filter_size=5, depth=3):
        self.depth, self.channels = depth, channels
        self.filter_size = filter_size

    def get_graph(self):
        def maker(inp):
            if len(inp.shape) < 3:
                inp = keras.layers.Lambda(K.expand_dims)(inp)
            for i in range(self.depth):
                inp = keras.layers.Conv1D(self.channels, self.filter_size, padding='same', activation='linear')(inp)
                inp = keras.layers.LeakyReLU(0.01)(inp)
                inp = keras.layers.BatchNormalization()(inp)
            return inp
        return maker


class SparseDNN(Neural):
    def __init__(self, size=512, depth=3, penalty_size=25, weight_percentage=0.1, typical_weight=0.01):
        self.depth = depth
        self.size = size
        self.penalty_size = 25
        self.weight_percentage = 0.1
        self.typical_weight = 0.01

    def get_graph(self):
        def maker(inp):
            if len(inp.shape) > 3:
                inp = keras.layers.Flatten()(inp)
            l1 = self.penalty_size / (self.weight_percentage * self.typical_weight * first.shape[-1] * self.size)
            for i in range(self.depth):
                inp = keras.layers.Dense(self.size, activation='linear', kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=0))(inp)
                inp = keras.layers.LeakyReLU(0.01)(inp)
                inp = keras.layers.BatchNormalization()(inp)
            return inp
        return maker


class AntimonotonyLayer(ToDo):
    pass


class CNN2D(Neural):
    def __init__(self, channels=32, filter_size=5, depth=3):
        self.depth, self.channels = depth, channels
        self.filter_size = filter_size

    def get_graph(self):
        def maker(inp):
            while len(inp.shape) > 4:
                inp = keras.layers.TimeDistributed(
                keras.layers.TimeDistributed(
                    keras.layers.Flatten()
                ))(inp)
            while len(inp.shape) < 4:
                inp = keras.layers.Lambda(K.expand_dims)(inp)
            for i in range(self.depth):
                inp = keras.layers.Conv2D(self.channels, self.filter_size, padding='same', activation='linear')(inp)
                inp = keras.layers.LeakyReLU(0.01)(inp)
                inp = keras.layers.BatchNormalization()(inp)
            return inp
        return maker


class SparseCNN1D(Neural):
    def __init__(self, channels=256, filter_size=5, depth=3, penalty_size=25, weight_percentage=0.1, typical_weight=0.01):
        self.depth, self.channels = depth, channels
        self.filter_size = filter_size
        self.penalty_size = 25
        self.weight_percentage = 0.1
        self.typical_weight = 0.01

    def get_graph(self):
        def maker(inp):
            while len(inp.shape) < 3:
                inp = keras.layers.Lambda(K.expand_dims)(inp)
            l1 = self.penalty_size / (self.weight_percentage * self.typical_weight * first.shape[1] * first.shape[2] * self.size)
            for i in range(self.depth):
                inp = keras.layers.Conv1D(self.channels, self.filter_size, padding='same', activation='linear', kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=0))(inp)
                inp = keras.layers.LeakyReLU(0.01)(inp)
                inp = keras.layers.BatchNormalization()(inp)
            return inp
        return maker


class RNN(Neural):
    """
    Recurrent RNN for last layer of the network
    """
    def __init__(self, width=512, depth=3, core=keras.layers.LSTM):
        self.width, self.depth, self.core = width, depth, core

    def get_graph(self):
        def maker(inp):
            while len(inp.shape) > 3:
                inp = keras.layers.TimeDistributed(keras.layers.Flatten())(inp)
            for i in range(self.depth):
                inp = self.core(self.width, return_sequences=True, kernel_initializer=Orthogonal())(inp)
                inp = keras.layers.BatchNormalization()(inp)
            return inp
        return maker


class Finalize(Neural):
    """
    Use this as a final stage of the recognition network
    """
    def __init__(self, phoneme_num, activation='softmax'):
        self.phoneme_num = phoneme_num
        self.activation = activation

    def new_network(self, dtype):
        first = inp = keras.layers.Input(shape=([None] + list(dtype.shape[1:])))
        if len(dtype.shape) > 2:
            first = keras.layers.Flatten()(first)
        outp = keras.layers.Dense(self.phoneme_num, activation=self.activation)
        mdl = Model(first, outp)
        return mdl


class Core(Neural):
    """
    Deprecated, left for tests sake
    """
    def __init__(self, width=512, depth=3):
        self.width, self.depth = width, depth

    def new_network(self, dtype):
        first = inp = keras.layers.Input(shape=[None, dtype.shape[1]])
        for i in range(self.depth):
            inp = keras.layers.GRU(self.width, activation='linear', return_sequences=True, kernel_initializer=Orthogonal())(inp)
            inp = keras.layers.LeakyReLU(0.01)(inp)
            inp = keras.layers.BatchNormalization()(inp)
        outp = inp
        mdl = Model(first, outp)
        return mdl

    def bind(self, other):
        pass


class SpectralCoefficients(ToDo):
    """
    Spectral Coefficients -> for MFCC
    """
    def spectral_coeffs(x):
        return dct(K.log(x**2 + 2e-12))[:, :, :20]
    
    
class FilterConstraint(keras.constraints.Constraint):
    def __init__(self, n_filters, n_fft):
        erb = TriangularERB(n_filts=n_filters, window=n_fft)
        self.mask = erb.filter != 0
        self.n_fft = n_fft
        self.n_filters = n_filters
    
    def __call__(self, v):
        if not hasattr(self, "mask_object"):
            self.mask_object = K.constant(self.mask)
        v = v * self.mask_object
        v = K.clip(v, 0, 1)
        return v
    
    def get_config(self):
        return {
            "n_filters": self.n_filters,
            "n_fft": self.n_fft,            
        }

_defaults["FilterConstraint"] = FilterConstraint


class LearnableFourierFBanks(Neural):
    """
    Initializes and adds constaints
    
    Need to register constraints to loaders...
    """
    
    def __init__(self, n_filters=64, fft_size=512):
        self.n_filters = n_filters
        self.fft_size = fft_size
        
    @property
    def n_fft(self):
        return self.fft_size // 2 + 1
        
    def new_network(self, dtype):
        constraint = FilterConstraint(
           self.n_filters, self.fft_size
        )
        first = inp = keras.layers.Input(shape=([None] + list(dtype.shape[1:])))
        if len(dtype.shape) > 2:
            first = keras.layers.Flatten()(first)
        lyr = keras.layers.Dense(self.n_filters, 
            use_bias=False,
            kernel_constraint=constraint
        )
        lyr = lyr(first)
        outp = keras.layers.BatchNormalization()(lyr)
        mdl = Model(inp, outp)
        return mdl
    


### Below are the aliases needed for the convenience of the dissertation


class TimeWarpingRNN(RNN):
    pass


class TimeWarpingCNN(CNN2D):
    pass


class EarlyDNN(DNN):
    pass


class EarlyConv2D(CNN2D):
    pass


class EarlyConv1D(CNN1D):
    pass


class EarlySparseDNN(SparseDNN):
    pass


class LaterConv1D(CNN1D):
    pass


class LaterDNN(DNN):
    pass


class LaterSparse1D(SparseCNN1D):
    pass


class Columns(Neural):
    """
    Joins neural layers
    """
    def __init__(self, feature_transforms, mapping):
        self.feature_transforms = [x for x in feature_transforms]
        self.mapping = mapping
        self.model_ref = None

    def get_graph(self):
        def maker(inp):
            columns = []
            for column in self.feature_transforms:
                next_item = inp
                for item in column:
                    next_item = item.get_graph()(next_item)
                columns.append(next_item)
            return keras.layers.Lambda(lambda x: self.mapping(self.model_ref, x))(columns)
        return maker

