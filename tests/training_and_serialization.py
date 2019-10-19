import keras
import scipy.io.wavfile as sio
import os
import unittest

from fwks.model import AcousticModel
from fwks.stage import RandomSelectionAdapter
import fwks.dataset as dataset
import fwks.stage as stage

from tests.meta import get_test_dataset


class TrainingTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.dset = get_test_dataset()

        def mk_model():
            inp = keras.layers.Input((None, 512))
            outp = keras.layers.Dense(38, activation='softmax')(inp)
            return keras.models.Model(inp, outp)
        
        self.am = AcousticModel([
            stage.Window(512, 512),
            stage.LogPowerFourier(),
            stage.MelFilterbank(20),
            stage.Core(width=512, depth=1),
            stage.CustomNeural(mk_model()),
            stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
        ])

        self.path = "/tmp/random_name.tmp.h5"  # not portable

    def test_01_training_acoustic_model(self):
        self.am.build(self.dset)
        self.am.summary()

    def test_02_saving_a_trained_model(self):
        self.am.save(self.path)

    def test_03_loading_a_trained_model(self):
        am = AcousticModel.load(self.path)
        am.summary()
        os.remove(self.path)
