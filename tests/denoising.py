import keras
import scipy.io.wavfile as sio
import os
import unittest

from fwks.model import DenoisingModel
from fwks.stage import RandomSelectionAdapter
import fwks.dataset as dataset
import fwks.stage as stage

from tests.meta import get_test_dataset


class DenoisingTest(unittest.TestCase):
    def test_01_czt(self):
        dset = get_test_dataset()
        
        def mk_model():
            inp = keras.layers.Input((None, 512))
            outp = keras.layers.Dense(38, activation='softmax')(inp)
            return keras.models.Model(inp, outp)

        am = DenoisingModel([
            stage.Window(512, 512),
            stage.LogPowerFourier(),
            stage.MelFilterbank(20),
            stage.Core(width=512, depth=1),
            stage.CustomNeural(mk_model()),
            stage.CTCLoss(selection_adapter=RandomSelectionAdapter())
        ])
        # To be done later
        # am.postprocessing = [
        #     stage.LogPowerFourier.reverse(),
        #     stage.OverlapAdd(512, 512)
        # ]
        am.build(dset)
