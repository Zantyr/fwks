import unittest

from fwks.tasks import FeatureLearnabilityTask, Task
from fwks import stage

from tests.meta import get_test_dataset

import keras


class MetricizationTestSuite(unittest.TestCase):
    """
    This test suite will:
    1. build a case where a model is build and PER is calculated
    2. build a case where a model is build in feature training and then feature quality is compared
    """
    
    def test_feature_learnability(self):
    
        FeatureLearnabilityTask.how_much = 100
        FeatureLearnabilityTask.epochs = 1

        def mfcc_model(preprocessing_distortion=None):
            return model_mod.AcousticModel(
                ([preprocessing_distortion] if preprocessing_distortion \
                    is not None else []) + [
                    stage.Window(512, 128),
                    stage.LogPowerFourier(),
                    stage.MelFilterbank(64),
                    stage.DCT(24),
                    stage.MeanStdNormalizer(),
                    stage.CNN2D(channels=16, filter_size=5, depth=2),
                    stage.RNN(width=512, depth=2),
                    stage.phonemic_map(37),
                    stage.CTCLoss()
                ], callbacks=[
                    keras.callbacks.TerminateOnNaN(),
                ])

        def small_network(self, source_shape, target_shape):
            source_shape = list(source_shape[1:])
            source_shape[0] = None
            mdl = keras.models.Sequential()
            mdl.add(keras.layers.Dense(512, activation='relu', input_shape=source_shape))
            mdl.add(keras.layers.Dense(512, activation='relu'))
            mdl.add(keras.layers.Dense(target_shape[-1]))
            return mdl
        
        class MFCCLearnability(metaclass=FeatureLearnabilityTask):
            @classmethod
            def get_mapping(self):
                return mfcc_model()

            @classmethod
            def get_dataset(cls):
                return get_test_dataset()
    
            get_mapper_network = classmethod(small_network)
        
        MFCCLearnability.run("/tmp/cache")
