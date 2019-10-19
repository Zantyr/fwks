import hashlib
from functools import reduce
import numpy as np
import operator
import os

from .stage_meta import SelectionAdapter, ToDo


class RandomSelectionAdapter(SelectionAdapter):
    """
    Divide recordings fully randomly. 
    This is generally the default selection adapter
    """
        
    def __init__(self, valid_percentage=0.1, test_percentage=0.1):
        self._train = self._valid = self._test = self._hash = None
        self.initialized = False
        self._valid_percentage = valid_percentage
        self._test_percentage = test_percentage

    def initialize(self, dataset):
        number = len(dataset.rec_fnames)
        train_threshold = 1. - self._valid_percentage - self._test_percentage
        valid_threshold = 1. - self._test_percentage
        selection = np.random.random(number)
        self._train = selection < train_threshold
        self._valid = (train_threshold < selection) & (selection <= valid_threshold)
        self._test = selection >= valid_threshold
        self._hash = hashlib.sha512(selection.tobytes()).digest().hex()[:16]

    @property
    def selection_hash(self):
        return self._hash
    
    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test
    
    def serialize(self):
        info = dict(valid_percentage=self.valid_percentage,
                    test_percentage=self.test_percentage)
        return RandomSelectionAdapter.build, info

    def build(self, info):
        return RandomSelectionAdapter(**info)
    

class SpeakerSelectionAdapter(SelectionAdapter):
    """
    Inherits: SelectionAdapter
    
    Then group according to speakers
    Then divide speakers in such a way as to fulfill the percentages
    """
    
    def __init__(self, valid_percentage=0.1, test_percentage=0.1):
        self._train = self._valid = self._test = self._hash = None
        self.initialized = False
        self._valid_percentage = valid_percentage
        self._test_percentage = test_percentage

    def initialize(self, dataset):
        speakers = [x for x in os.listdir(dataset.root) if x.startswith("SES") and os.path.isdir(os.path.join(dataset.root, x))]
        per_speaker = [[x for x in dataset.rec_fnames if speaker in x] for speaker in speakers]
        selection = np.random.random(len(speakers))
        train_threshold = 1. - self._valid_percentage - self._test_percentage
        valid_threshold = 1. - self._test_percentage
        self._train = selection < train_threshold
        self._valid = (train_threshold < selection) & (selection <= valid_threshold)
        self._test = selection >= valid_threshold
        select_per_speaker = lambda selected: [dataset.rec_fnames.index(x) for x in reduce(operator.add, [per_speaker[ix] for ix, x in enumerate(selected) if x])]
        self._train = select_per_speaker(self._train)
        self._valid = select_per_speaker(self._valid)
        self._test = select_per_speaker(self._test)
        self._hash = hashlib.sha512(selection.tobytes()).digest().hex()[:16]

    @property
    def selection_hash(self):
        return self._hash
    
    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test
    
    def serialize(self):
        info = dict(valid_percentage=self.valid_percentage,
                    test_percentage=self.test_percentage)
        return RandomSelectionAdapter.build, info

    def build(self, info):
        return RandomSelectionAdapter(**info)

    
    
    