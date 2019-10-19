"""
Things that do not fit elsewhere
"""

from .stage_meta import ToDo, Analytic, DType
from functools import reduce

import numpy as np


class Pointwise(ToDo):
    """
    Squash and so on
    """

    
class LogPower(Analytic):
    def __init__(self, negative=True):
        self.negative = negative
    
    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return input_dtype
    
    def _function(self, recording):
        return (-1 if self.negative else 1) * np.log(np.abs(recording))
    
    
class ComposeFeatures(Analytic):
    def __init__(self, feature_transforms):
        self.feature_transforms = feature_transforms
        
    def bind(self, prev):
        self.previous = prev
        prev = None
        for i in self.feature_transforms:
            prev = i.bind(prev)
        return self

    def output_dtype(self, input_dtype):
        for transform in self.feature_transforms:
            input_dtype = transform.output_dtype(input_dtype)
        return input_dtype

    def _function(self, recording):
        return self.feature_transforms[-1].map(recording)
        
        
    
class ConcatFeatures(Analytic):
    def __init__(self, feature_transforms, max_fit = 10):
        self.feature_transforms = [(ComposeFeatures(x) if isinstance(x, list) else x)
            for x in feature_transforms]
        self.max_fit = max_fit
        
    def bind(self, prev):
        self.previous = prev
        [transform.bind(None)
            for transform in self.feature_transforms]
        return self
    
    def output_dtype(self, input_dtype):
        dtypes = [print(input_dtype) or transform.output_dtype(input_dtype)
            for transform in self.feature_transforms]
        shape = sum([dtype.shape[-1] for dtype in dtypes])
        shape = dtypes[0].shape[:-1] + [shape]
        print(dtypes, shape, input_dtype)
        return DType("Array", shape, np.float32)
    
    def _function(self, recording):
        transforms = [transform._function(recording)
            for transform in self.feature_transforms]
        times = np.array([x.shape[0] for x in transforms])
        times -= times.min()
        if times.max() < self.max_fit:
            max_time = np.array([x.shape[0] for x in transforms]).max()
            transforms = [
                np.pad(x, tuple([(0, max_time - x.shape[0])] + [
                    (0, 0) for dim in x.shape[1:]
                ]), 'constant') for x in transforms
            ]
        transforms = np.concatenate(transforms, axis=(len(transforms[0].shape) - 1))
        return transforms
