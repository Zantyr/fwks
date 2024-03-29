import abc
import functools
import itertools
import operator
import os
import scipy.io.wavfile as sio
import tqdm


class LoaderAdapter(metaclass=abc.ABCMeta):

    returns_transcripts = True

    @property
    def loader_function(self):
        """
        If overridden, should accept filename and return (sample_rate, data)
        """
        return sio.read

    def get_recording_lengths(self):
        lens = []
        print("Getting dataset lengths")
        for f, _ in tqdm.tqdm(itertools.zip_longest(*self.get_fnames())):
            sr, data = sio.read(f)
            try:
                assert sr == self.accepted_sr
                if self.n_channels == 1:
                    assert len(data.shape) == 1
                    lens.append(len(data))
                else:
                    data.shape[0] == self.n_channels
                    lens.append(data.shape[1])
                # TODO: FIX: when ignoring bad SR, also should modify the filenames...
            except AssertionError:
                if self.throw_on_mismatch:
                    raise
        return lens

    @abc.abstractmethod
    def get_fnames(self):
        return rec_fnames, trans_fnames


class PlainAdapter(LoaderAdapter):
    """
    Default LoaderAdapter. Loads data from a single directory.
    Every WAVE is loaded as a source recording and matches text files are treated
    as transcriptions.
    """

    def __init__(self, root, accepted_sr=16000, throw_on_mismatch=True, n_channels=1):
        """
        accepted_sr - which sampling rate should the recordings have? [Hz]
        throw_on_mismatch - whether to raise errors in case of bad sampling rate or to simply ignore that recording
        n_channels - How many channels should the recording have?
        """
        self.ROOT = root
        self.accepted_sr = accepted_sr
        self.throw_on_mismatch = throw_on_mismatch
        self.n_channels = n_channels

    def get_fnames(self):
        rec_fnames, trans_fnames = [], []
        for i in os.listdir(self.ROOT):
            if i.endswith(".wav"):
                rec_fnames.append(os.path.join(self.ROOT, i))
                txt = os.path.join(self.ROOT, i.replace(".wav", ".txt"))
                if txt not in trans_fnames:
                    trans_fnames.append(txt)
        return rec_fnames, trans_fnames


class ClarinAdapter(LoaderAdapter):
    """
    Loads data from Polish Clarin dataset.
    """

    def __init__(self, root, accepted_sr=16000, throw_on_mismatch=True, n_channels=1):
        self.ROOT = root
        self.accepted_sr = accepted_sr
        self.throw_on_mismatch = throw_on_mismatch
        self.n_channels = n_channels


    def get_fnames(self):
        rec_fnames, trans_fnames = [], []
        for i in [x for x in os.listdir(self.ROOT) if os.path.isdir(os.path.join(self.ROOT, x))]:
            recordings = os.path.join(self.ROOT, i, "wav")
            transcripts = os.path.join(self.ROOT, i, "lab")
            for fname in os.listdir(recordings):
                core, extension = fname.split(".")
                assert extension == "wav"
                if os.path.isfile(os.path.join(transcripts, core + ".hlb")):
                    rec_fnames.append(os.path.join(recordings, fname))
                    trans_fnames.append(os.path.join(transcripts, core + ".hlb"))
        return rec_fnames, trans_fnames

    @staticmethod   # added, if this fails - debug there
    def get_phones_clarin(fname):
        with open(fname, "r", encoding="utf-8") as f:
            s = f.read()
            s = s.split("Phoneme Phoneme")[1]
            s = s.split("\n\n")[0]
            s = [x.split(' ')[1] for x in s.split('\n') if x.strip()]
            s = [x.split('_')[0] for x in s]
        return s

    @staticmethod
    def get_words_clarin(fname):
        with open(fname, encoding="utf-8") as f:
            text = f.read()
        a = text.split('Word Word')[1].split('\n\n')[0].split('\n')
        a = [x.strip() for x in a if x.strip()]
        a = [x.split()[1] for x in a]
        return a


class MappingAdapter:
    @abc.abstractproperty
    def produces(self):
        pass

    @abc.abstractmethod
    def generate_requirement(self, dataset, mapping, req):
        pass


class MixturesMapper(MappingAdapter):
    """
    Takes clean recordings and produces mixtures of random selection of recordings for separation.
    """

    def __init__(self, n_mixtures):
        """
        n_mixtures - number of recordings in single n_mixtures
        """
        self.n_mixtures = n_mixtures

    def generate_requirement(self, dataset, mapping, req):
        for i in range():
            pass


        print("Getting clean recordings")
        n_recs = len(self.rec_fnames)
        dtype = mapping.output_dtype(stage.DType("Array", [max(self.recording_lengths)], np.float32))
        recordings = np.zeros([n_recs] + dtype.shape, dtype.dtype)
        lens = []
        for ix, fname in enumerate(tqdm.tqdm(self.rec_fnames)):
            sr, data = self._loader_adapter.loader_function(fname)
            assert sr == self.sr
            data = data.astype(np.float32) / 2**15
            data = mapping.map(data)
            lens.append(data.shape[0])
            key = [ix] + [slice(None, x, None) for x in data.shape]
            recordings.__setitem__(key, data)
        if hasattr(mapping, "normalize"):
            if not (mapping.trained if hasattr(mapping, "trained") else hasattr(mapping, "mean")):
                print("Applying normalization")
                recordings = mapping.normalize(recordings, lens)
        self.clean = recordings
        self.clean_lens = np.array(lens)





    @property
    def produces(self):
        return ["mixture"] + ["component_{}".format(x + 1) for x in range(self.n_mixtures)]

    def _get_mixture(self):
        rec_1 = self.from_cache() # indices from randomization
        rec_2 = self.from_cache()
        recordings = []
        for i in selection:
            recordings.append(selecasdfghklhgfd)
        return functools.reduce(operator.add, recordings) # simple additive mixture
