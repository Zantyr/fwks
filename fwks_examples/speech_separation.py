"""
Speech separation task test
"""

import unittest

from fwks import stage


from keras import layers

import keras
import tensorflow as tf
import keras.backend as K

from fwks.tasks import make_training_task
from fwks.model import DenoisingModel
from fwks.dataset import MixturesMapper, Dataset


class SpeechSeparation(unittest.TestCase):
    def custom_neural(self):
        first = layers.Input(shape=(None, 257))
        lyr = first
        lyr = layers.Dense(512, activation='sigmoid')(lyr)
        lyr = layers.Convolution1D(kernel_size=5, filters=512, activation='relu')(lyr)
        lyr = layers.Convolution1D(kernel_size=5, filters=512, activation='relu')(lyr)
        lyr = layers.LSTM(512, return_sequences=True)(lyr)
        lyr = layers.LSTM(512, return_sequences=True, activation='sigmoid')(lyr)
        lyr = layers.Lambda(lambda x: K.stack([x[:, :, :256], x[:, :, 256:]], axis=-1))(lyr)
        lyr = layers.Lambda(lambda x: tf.pad(x, ((0, 0), (0, 0), (1, 0), (0, 0))))(lyr)
        lyr = layers.Lambda(lambda x: x[0] * K.stack([x[1], x[1]], axis=-1) )([lyr, first])
        return keras.models.Model(first, lyr)

    def test_model_make(self):
        mdl = self.custom_neural()
        mdl.summary()

    def test_speech_separation(self):
        am = DenoisingModel([
            stage.Window(512, 256),
            stage.LogPowerFourier(),
            stage.CustomNeural(self.custom_neural),
            stage.PermutationInvariantLoss(
                stage.L2Loss()  # add mapping
            )
        ])
        
        TaskBase = make_training_task()

        class SpeechSeparation(metaclass=TaskBase):

            how_much = 32
            epochs = 1
            batch_size = 8
        
            @classmethod
            def get_dataset(cls):
                dset = Dataset()
                dset.loader_adapter = "plain"
                dset.mapping = [MixturesMapper(2)]
                dset.get_from("tests/test_data")
                return dset
        
            @classmethod
            def get_acoustic_model(cls):
                return am

        print(dir(SpeechSeparation))
        SpeechSeparation.run("/tmp/fwks_cache")


if __name__ == "__main__":
    unittest.main()