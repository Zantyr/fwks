import dill
import keras
import numpy as np
import os
import shutil
import tempfile
import time
import zipfile

import keras.backend as K
import tensorflow as tf

from fwks.stage import Neural, Analytic, Loss, DType
from .meta import Representation, ItemLoader, SoundModel, MappingGenerator
from fwks.miscellanea import PeekGradients, StopOnConvergence

try:
    import pynini
except ImportError:
    import warnings
    warnings.warn("Cannot import pynini")

_defaults = {"tf": tf}    # this should contain everything...


class AcousticModel(SoundModel):

    def predict(self, recording, literal=True, number_of_preds=1, beam_width=2500):
        predictions = self.predict_raw(recording)
        decoded = K.ctc_decode(predictions, [predictions.shape[1]] * predictions.shape[0], greedy=False, beam_width=beam_width, top_paths=number_of_preds)
        if literal:
            # print(decoded)
            all_translations = []
            for recording in (decoded[0][x] for x in range(len(decoded[0]))):
                recording = recording.eval(session=K.get_session())
                # print(recording)
                rec_translations = []
                for attempt in recording:
                    rec_translations.append([self.symbol_map[x] for x in attempt])
                all_translations.append(rec_translations)
            return all_translations
        else:
            return decoded

    def to_wfst(self, recording):
        phonemes = self.predict_raw(recording)
        EPSILON = 0
        fst = pynini.Fst()
        init = fst.add_state()
        fst.set_start(init)
        heads = [(init, EPSILON)]
        num_of_letters = phonemes.shape[2]
        time = phonemes.shape[1]
        letters = [x+1 for x in range(num_of_letters)]
        for time in range(time):
            states = [fst.add_state() for _ in letters]
            log_phonemes = -np.log(phonemes[0])
            for entering_state, head in heads:
                for letter, letter_state in zip(letters, states):
                    if letter == len(letters):
                        letter = 0
                    # letter_state = fst.add_state()
                    output_sign = head if head != letter else 0
                    weight = log_phonemes[time, letter]
                    fst.add_arc(entering_state, pynini.Arc(
                        letter, output_sign, weight, letter_state))
            heads = list(zip(states, letters))
        [fst.set_final(x[0]) for x in heads]
        if optimize:
            fst.optimize()
        return fst

    def __str__(self):
        if self.built:
            return "<Trained acoustic model with loss {}>".format(self.statistics["loss"])
        else:
            return "<Untrained acoustic model>"

    __repr__ = __str__

    def summary(self, show=True):
        if self.built:
            statstring = "\n    ".join(["{}: {}".format(k, v) for k, v in self.statistics.items()])
            docstring = ("--------\nTrained acoustic model named \"{}\"\nDataset signature: {}\n"
                         "Dataset train-valid-test selector signature: {}\nTraining time: {}\n"
                         "Model complexity: {}\nStatistics:\n"
                         "    {}\n--------").format(self.name, self.dataset_signature, self.split_signature, self.building_time, self.complexity, statstring)
        else:
            docstring = "Untrained acoustic model named \"{}\"".format(self.name)
        if show:
            print(docstring)
        else:
            return docstring
