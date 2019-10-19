import keras
import numpy as np


class PeekGradients(keras.callbacks.Callback):
    """
    Used to debug NaNs
    """
    def on_batch_end(self, batch, logs=None):
        print("Layer weight statistics")
        for lyr in self.model.layers:
            print(lyr.name)
            for tensor in lyr.get_weights():
                print(tensor.mean(), "+-", tensor.std())


class StopOnConvergence(keras.callbacks.Callback):
    def __init__(self, max_repetitions=10):
        super().__init__()
        self.max_repetitions = max_repetitions

    def on_train_begin(self, logs=None):
        self.repetitions = 0
        self.last_loss = np.inf

    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('val_loss')
        if loss is not None:
            if loss > self.last_loss:
                self.repetitions += 1
            else:
                self.last_loss = loss
                self.repetitions = 0
            if self.repetitions > self.max_repetitions:
                self.model.stop_training = True
