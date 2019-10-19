import keras
import keras.backend as K
from keras.layers import Lambda, Input
from keras.models import Model
import numpy as np

from .stage_meta import Loss
from .stage_selection_adapter import SpeakerSelectionAdapter


class CTCLoss(Loss):
    def __init__(self, optimizer=None, use_noisy=False, selection_adapter=None):
        self.optimizer = optimizer if optimizer else keras.optimizers.Adam(clipnorm=1.)
        self.use_noisy = use_noisy
        self.selection_adapter = selection_adapter if selection_adapter else SpeakerSelectionAdapter()

    def compile(self, network, callbacks=[]):
        label_input = Input(shape = (None,))
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_lambda = Lambda(lambda args:K.ctc_batch_cost(*args), output_shape=(1,), name='ctc')([label_input, network.outputs[0], input_length, label_length])
        model = Model([network.inputs[0], label_input, input_length, label_length], [loss_lambda])
        model.compile(loss=(lambda y_true, y_pred: y_pred), optimizer=self.optimizer)
        return model

    @property
    def selection_hash(self):
        return self.selection_adapter.selection_hash

    @property
    def requirements(self):
        if self.use_noisy:
            return ["noisy", "transcripts"]
        return ["clean", "transcripts"]

    def fetch_train(self, dataset):
        self.selection_adapter.initialize(dataset)
        selection = self.selection_adapter.train
        if self.use_noisy:
            return [[
                dataset.noisy[selection],
                dataset.transcriptions[selection],
                dataset.noisy_lens[selection],
                dataset.transcription_lens[selection]
            ], np.zeros(dataset.noisy[selection].shape[0])]
        return [[
            dataset.clean[selection],
            dataset.transcriptions[selection],
            dataset.clean_lens[selection],
            dataset.transcription_lens[selection]
        ], np.zeros(dataset.clean[selection].shape[0])]

    def fetch_valid(self, dataset):
        self.selection_adapter.initialize(dataset)
        selection = self.selection_adapter.valid
        if self.use_noisy:
            return [[
                dataset.noisy[selection],
                dataset.transcriptions[selection],
                dataset.noisy_lens[selection],
                dataset.transcription_lens[selection]
            ], np.zeros(dataset.noisy[selection].shape[0])]
        return [[
            dataset.clean[selection],
            dataset.transcriptions[selection],
            dataset.clean_lens[selection],
            dataset.transcription_lens[selection]
        ], np.zeros(dataset.clean[selection].shape[0])]

    def fetch_test(self, dataset):
        self.selection_adapter.initialize(dataset)
        selection = self.selection_adapter.test
        if self.use_noisy:
            return [[
                dataset.noisy[selection],
                dataset.transcriptions[selection],
                dataset.noisy_lens[selection],
                dataset.transcription_lens[selection]
            ], np.zeros(dataset.noisy[selection].shape[0])]
        return [[
            dataset.clean[selection],
            dataset.transcriptions[selection],
            dataset.clean_lens[selection],
            dataset.transcription_lens[selection]
        ], np.zeros(dataset.clean[selection].shape[0])]

    def serialize(self):
        info = dict(optimizer=self.optimizer.__class__,
                    optimizer_config=self.optimizer.get_config(),
                    use_noisy=self.use_noisy,
                    selection_adapter=self.selection_adapter)
        return CTCLoss.builder, info

    @staticmethod
    def builder(info):
        optimizer = info["optimizer"].from_config(info["optimizer_config"])
        return CTCLoss(
            optimizer=optimizer,
            use_noisy=info["use_noisy"],
            selection_adapter=info["selection_adapter"]
        )



class L2Loss(Loss):
    def __init__(self, optimizer=None, selection_adapter=None):
        self.optimizer = optimizer if optimizer else keras.optimizers.Adam(clipnorm=1.)
        self.selection_adapter = selection_adapter if selection_adapter else SpeakerSelectionAdapter()

    def compile(self, network, callbacks=[]):
        model = Model(network.inputs[0], network.inputs[0])
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    @property
    def selection_hash(self):
        return self.selection_adapter.selection_hash

    @property
    def requirements(self):
        return ["clean", "noisy"]

    def fetch_train(self, dataset):
        self.selection_adapter.initialize(dataset)
        selection = self.selection_adapter.train
        return [dataset.noisy[selection],
                dataset.clean[selection]]

    def fetch_valid(self, dataset):
        self.selection_adapter.initialize(dataset)
        selection = self.selection_adapter.valid
        return [dataset.noisy[selection],
                dataset.clean[selection]]

    def fetch_test(self, dataset):
        self.selection_adapter.initialize(dataset)
        selection = self.selection_adapter.test
        return [dataset.noisy[selection],
                dataset.clean[selection]]

    def serialize(self):
        info = dict(optimizer=self.optimizer.__class__,
                    optimizer_config=self.optimizer.get_config(),
                    selection_adapter=self.selection_adapter)
        return L2Loss.builder, info

    @staticmethod
    def builder(info):
        optimizer = info["optimizer"].from_config(info["optimizer_config"])
        return L2Loss(
            optimizer=optimizer,
            selection_adapter=info["selection_adapter"]
        )
