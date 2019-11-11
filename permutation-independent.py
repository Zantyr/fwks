# https://arxiv.org/pdf/1607.00325.pdf

# Dong Yu


# https://github.com/pchao6/LSTM_PIT_Speech_Separation




# CASA - pre DL: # CASA - pre DL:


Generally use the minimal separation

LSTM
LSTM
K.stack

# separate LSTMs or bridged?

# augmentation

[case, time, freq]
[case, time, freq, channel]


import itertools


def pit_training_factory(n_sources, loss):
  permutations = zip(
        itertools.repeat(range(n_sources)),
        itertools.permutations(range(n_sources))
  )
  def final_loss(true, targets):
    losses = []
    for permutation in permutations:
      loss_components = []
      for i, j in permutation:

        loss_components.append(
          loss(true[:, :, :, i], targets[:, :, :, j])
        )
      losses.append(K.sum(loss_components))
    return K.min(losses)

  return final_loss


class Mixtures(MappingAdapter):
    """
    .produces

    Produces "mixed, mixture_1, mixture_2, mixture_..."
    What is the call
    """
    def __init__(self, n_mixtures):
        self.n_mixtures = n_mixtures

    def __getattr__(self, k):
        return super().__gettattr__(k)

    def get_clean(self, clean):
        pass

    def _from_cache(self):
        """
        if possible: load from this if possible
        """

    def _get_mixture():
        rec_1 = ...
        rec_2 = ...
        return rec_1 + rec_2   # simple additive mixture



am = AcousticModel([
    stage.Window(512, 256),  #
    stage.LogPowerFourier(),
    stage.KerasSequential([
        layers.Dense(512, activation='sigmoid'),
        layers.Convolutional1D(kernel=5, filters=1024, activation='relu'),
        layers.Convolutional1D(kernel=5, filters=1024, activation='relu'),
        layers.LSTM(1024),
        layers.LSTM(512),
        layers.Lambda(lambda x: K.stack([x[:, :, :256], x[:, :, 256:]], axis=-1))
    ]),
    stage.PermutationInvariantTraining(

    )
])

am.mapping = [Mixtures(2)]
am.build(dset)
