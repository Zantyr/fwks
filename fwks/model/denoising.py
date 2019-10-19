from .meta import MappingGenerator, SoundModel


class DenoisingModel(SoundModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._postprocessing = None

    @property
    def postprocessing(self):
        return self._postprocessing

    @postprocessing.setter
    def postprocessing(self, v):
        self._postprocessing = MappingGenerator(self.stages).get(v)

    def predict(self, input):
        recording = self.predict_raw(input)
        dtype = self.mapping.output_dtype(DType("Array", recording.shape[1:], recording.dtype))
        output = np.zeros([recording.shape[0]] + dtype.shape, dtype=dtype.dtype)
        for i in range(recording.shape[0]):
            output[i] = self._postprocessing.map(recording[i])
        return output

    def __str__(self):
        if self.built:
            return "<Trained acoustic model with loss {}>".format(self.statistics["loss"])
        else:
            return "<Untrained acoustic model>"

    __repr__ = __str__

    def summary(self, show=True):
        if self.built:
            statstring = "\n    ".join(["{}: {}".format(k, v) for k, v in self.statistics.items()])
            docstring = ("--------\nTrained denoising model named \"{}\"\nDataset signature: {}\n"
                         "Dataset train-valid-test selector signature: {}\nTraining time: {}\n"
                         "Model complexity: {}\nStatistics:\n"
                         "    {}\n--------").format(self.name, self.dataset_signature, self.split_signature, self.building_time, self.complexity, statstring)
        else:
            docstring = "Untrained denoising model named \"{}\"".format(self.name)
        if show:
            print(docstring)
        else:
            return docstring
