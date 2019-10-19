from .stage_meta import Normalizer

import numpy as np


class MeanStdNormalizer(Normalizer):
    def normalize(self, features, lengths):
        count = 0
        self.mean = 0
        self.std = 0
        for rec, rec_len in zip(features, lengths):
            count += rec_len
            self.mean += rec_len * rec[:rec_len].mean()
        self.mean /= count
        for rec, rec_len in zip(features, lengths):
            self.std += rec_len * ((rec[:rec_len] - self.mean) ** 2).mean()
        self.std /= count - 1
        self.std = np.sqrt(self.std)
        print("Featural statistics: {} +- {}".format(self.mean, self.std))
        returnable = (features - self.mean) / self.std
        print("Normalized to mean and std: {} +- {}".format(returnable.mean(), returnable.std()))
        return returnable
    
    def map(self, recording):
        if self.previous is not None:
            recording = self.previous.map(recording)
        if hasattr(self, "mean"):
            return (recording - self.mean) / self.std
        return recording