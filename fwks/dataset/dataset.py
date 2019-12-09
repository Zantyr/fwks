import hashlib
import numpy as np
import os
import random
import scipy.io.wavfile as sio
from fwks import stage
import tqdm

from .adapters import MappingAdapter, LoaderAdapter, PlainAdapter, ClarinAdapter


# move to miscellanea
class PermutedView:
    def __init__(self, data, mapping, setter=True):
        self.data = data
        if isinstance(mapping, list):
            self._mapping_list = mapping
            self.mapping = lambda number: self._mapping_list[number]
        else:
            self._mapping_list = None
            self.mapping = mapping

    def __getitem__(self, k):
        return self.data[self.mapping(k)]

    def __setitem__(self, k, v):
        if self.setter:
            self.data[self.mapping(k)] = v
        raise RuntimeError("Cannot set values to this PermutedView")


def get_recording_lengths(fnames):
    lens = []
    print("Getting dataset lengths")
    for f in tqdm.tqdm(fnames):
        sr, data = sio.read(f)
        assert sr == 16000
        assert len(data.shape) == 1
        lens.append(len(data))
    return lens

def get_phones_clarin(fname):
    with open(fname, "r", encoding="utf-8") as f:
        s = f.read()
        s = s.split("Phoneme Phoneme")[1]
        s = s.split("\n\n")[0]
        s = [x.split(' ')[1] for x in s.split('\n') if x.strip()]
        s = [x.split('_')[0] for x in s]
    return s

def _generate_fundamental_pair(frequency, n_harms=6, sr=16000, length=160000):
    t = np.arange(length).astype(np.float32)
    t = t / sr * 2 * np.pi
    with_base = np.zeros([length], np.float32)
    without_base = np.zeros([length], np.float32)
    powers = np.random.random(n_harms - 1) + 1
    base_power = np.min(powers)
    with_base += base_power * np.sin(t * frequency)
    for i, power in enumerate(powers):
        without_base += power * np.sin(t * frequency * (i + 2))
        with_base += power * np.sin(t * frequency * (i + 2))
    return with_base, without_base

def _fundamental_freqs(how_long, sr):
    freq = np.random.randint(20, 500)
    n_harms = np.random.randint(4, 9)
    return _generate_fundamental_pair(freq, n_harms, sr=sr, length=how_long)


class AbstractDataset:
    def from_cache(self, k):
        """
        if possible: load from this if possible

        load
        # should be freed afterwards if not used
        """
        if not hasattr(self, "_clean_cache"):
            self._clean_cache = [None for i in range(len(self.rec_fnames))]
        if self._clean_cache[k]:
            sr, data = self._loader_adapter.loader_function(self.rec_fnames[k])
            assert sr == self.sr
            data = data.astype(np.float32) / 2**15
            self._clean_cache[k] = data
        return self._clean_cache[k]


class Dataset(AbstractDataset):
    """
    Dataset represents a collection of recordings, transcripts and other data constituting corpora
    for building and evaluation of the models. This class should aggregate all data that is required
    for those tasks. Adapters specify the way in which to load and prepare the recordings.

    Dataset most often does not store the clean recordings, instead relying on
    storing the transformed representation. The transform is fetched from the model specification.
    In this way, the datasets are never complete without the accompanying model, which specifies
    what is the interpretation of the data.

    Dataset can be seen as key-value mapping between forms of data and the corpora proper. Each value
    is a collection of items, each item is a piece of data corresponding to other items from each key at
    the same index. E.g. dataset contains a collection of 320 clean recordings in STFT form in form of
    a numpy array with shape (320, _, _) and a list of transcriptions, length of list being 320 and each
    item in the list is another list of strings, each string being a single word.
    """

    _basic_accepted_requirements = ["clean", "transcripts", "noisy", "stft"]
    __adapters = {
        "plain": PlainAdapter,
        "clarin": ClarinAdapter
    }
    __mapping_adapters = {}

    def __init__(self, noise_gen=None, sr=16000):
        """
        noise_gen - if applicable, what noise generator to use to produce noisy recordings; no value will raise an Error when attempting to fetch noisy recordings
        sr - sampling rate of the recordings
        """
        self.sr = sr
        self.root = None
        self.all_phones = None
        self.recording_lengths = None
        self.rec_fnames = None
        self.trans_fnames = None
        self.transcriptions = None
        self.transcription_lens = None
        self.clean = None
        self.clean_lens = None
        self.noisy = None
        self.noisy_lens = None
        self._hash = None
        self.noise_gen = noise_gen
        self._loader_adapter = PlainAdapter
        self._mapping_adapters = []

    @property
    def _accepted_requirements(self):
        from_adapters = []
        for adapter in self._mapping_adapters:
            from_adapters += adapter.produces
        return self._basic_accepted_requirements + from_adapters

    def get_from(self, root):
        self.root = root
        self._loader_adapter = self._loader_adapter(self.root)
        rec_fnames, trans_fnames = self._loader_adapter.get_fnames()
        lens = self._loader_adapter.get_recording_lengths()
        sorted_records = np.array(sorted(list(range(len(lens))), key=lambda ix: lens[ix]))
        rec_fnames = [rec_fnames[x] for x in sorted_records]
        lens = [lens[x] for x in sorted_records]
        self.rec_fnames = rec_fnames
        self.recording_lengths = lens
        if self._loader_adapter.returns_transcripts:
            trans_fnames = [trans_fnames[x] for x in sorted_records]
            self.trans_fnames = trans_fnames

    def generate(self, mapping, requirements):
        if isinstance(self._loader_adapter, MappingAdapter):
            _accepted_requirements = self._accepted_requirements + self._loader_adapter.produces
            _mapping_adapters = self._mapping_adapters + [self._loader_adapter]
        else:
            _mapping_adapters = self._mapping_adapters
        assert all([x in _accepted_requirements for x in requirements]), \
            "Requirements of the model are not satisfied. " \
            "Required items: {}, producable: {}".format(requirements, self._accepted_requirements)
        for req in requirements:
            if req == "clean":
                self._get_cleans(mapping)
            elif req == "transcripts":
                self._get_transcripts()
            elif req == "noisy":
                self._get_noisy(mapping)
            elif req == "stft":
                self._get_stft()
            else:
                for adapter in _mapping_adapters:
                    if req in adapter.produces:
                        value = adapter.generate_requirement(self, mapping, req)
                        setattr(self, req, value)
                        break
                else:
                    raise ValueError("Requirement {} is not produced by any of the adapters".format(req))
        if hasattr(self, "_clean_cache"):
            del self._clean_cache
        self._hash = hashlib.sha512(str(self.rec_fnames).encode("utf-8")).digest().hex()[:16]

    def select_first(self, count):
        self.select_slice(None, count)

    def select_indices(self, selection):
        self.rec_fnames = [self.rec_fnames[x] for x in selection]
        self.trans_fnames = [self.trans_fnames[x] for x in selection]
        self.recording_lengths = [self.recording_lengths[x] for x in selection]

    def select_random(self, count):
        selection = random.sample(range(len(self.rec_fnames)), count)
        self.select_indices(selection)

    def select_slice(self, start, end=None, step=None):
        if isinstance(start, slice):
            selection = start
        else:
            selection = slice(start, end, step)
        self.rec_fnames = self.rec_fnames[selection]
        self.trans_fnames = self.trans_fnames[selection]
        self.recording_lengths = self.recording_lengths[selection]

    def generate_dtype(self, mapping):
        dtype = stage.DType("Array", [max(self.recording_lengths)], np.float32)
        return mapping.output_dtype(dtype)

    @property
    def signature(self):
        return self._hash

    def cache_choice(self):
        pass

    def _get_transcripts(self):
        print("Getting list of phones")
        transes = (get_phones_clarin(x) for x in tqdm.tqdm(self.trans_fnames))
        self.all_phones = set()
        def len_add_set(trans):
            self.all_phones |= set(trans)
            return len(trans)
        trans_lens = [len_add_set(x) for x in transes]
        self.all_phones = list(self.all_phones)
        phone_zero = len(self.all_phones)
        print("Getting transcriptions")
        phones_shape = [len(trans_lens), max([x for x in trans_lens]) + 1]
        transes = np.full(phones_shape, phone_zero, np.uint16)
        for num, fname in tqdm.tqdm(enumerate(self.trans_fnames)):
            trans = get_phones_clarin(fname)
            trans = np.array([self.all_phones.index(x) for x in trans])
            transes[num, :len(trans)] = trans
        self.transcriptions = transes
        self.transcription_lens = np.array(trans_lens)

    def _get_cleans(self, mapping):
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

    def _get_noisy(self, mapping):
        print("Getting distorted recordings")
        n_recs = len(self.rec_fnames)
        dtype = mapping.output_dtype(stage.DType("Array", [max(self.recording_lengths)], np.float32))
        recordings = np.zeros([n_recs] + dtype.shape, dtype.dtype)
        lens = []
        for ix, fname in enumerate(tqdm.tqdm(self.rec_fnames)):
            sr, data = sio.read(fname)
            assert sr == 16000
            data = data.astype(np.float32) / 2**15
            data = self.noise_gen.pre(data)
            data = mapping.map(data)
            data = self.noise_gen.post(data)
            lens.append(data.shape[0])
            key = [ix] + [slice(None, x, None) for x in data.shape]
            recordings.__setitem__(key, data)
        if hasattr(mapping, "normalize"):
            if not mapping.trained:
                print("Applying normalization")
                recordings = mapping.normalize(recordings, lens)
        self.noisy = recordings
        self.noisy_lens = np.array(lens)

    def get_metrics(self, print=True):
        pass

    def calculate_metric(self, recording, transcription):
        pass

    def _get_stft(self):
        print("Getting standard STFT")
        n_recs = len(self.rec_fnames)
        frames = max(self.recording_lengths) // 128 - 3
        recordings = np.zeros([n_recs, frames, 257], np.float32)
        for ix, fname in enumerate(tqdm.tqdm(self.rec_fnames)):
            sr, data = sio.read(fname)
            assert sr == 16000
            data = data.astype(np.float32) / 2**15
            for time in range(self.recording_lengths[ix] // 128 - 3):
                recordings[ix, time, :] = np.log(np.fft.rfft(data[time * 128 : time * 128 + 512]) ** 2 + 2e-12)
        self.stft = recordings

    @property
    def loader_adapter(self):
        return self._loader_adapter

    @loader_adapter.setter
    def loader_adapter(self, v):
        if isinstance(v, str):
            if not v in self.__adapters.keys():
                raise TypeError("Key {} not in default adapters".format(v))
            self._loader_adapter = self.__adapters[v]
        elif not issubclass(v, LoaderAdapter):
            print(v, LoaderAdapter, issubclass(v, LoaderAdapter))
            raise TypeError("{} is not a proper Adapter; it should inherit from Adapter class".format(v))
        else:
            self._loader_adapter = v

    # hotfix accessor
    @property
    def transcriptions_lens(self):
        return self.transcription_lens


class SyntheticDataset(AbstractDataset):
    """
    Synthetic dataset represents a collection of data that is generated without underlying files.
    The data serves best for testing the models or training a specific kind of transform.
    This cannot generate realistic speech in general.
    """

    _functions = {
        "fundamental_freqs": _fundamental_freqs
    }

    def __init__(self,
                 fn="fundamental_freqs",
                 how_much=320,
                 how_long=160000,
                 what_is_generated=None,
                 sr=16000):
        """
        fn - what function to use, either function type or string matching keys in SyntheticDataset._functions
        how_much - self-describing
        how_long - in samples
        what_is_generated - specifies keys, by which the generated data from fn can be queried by the models
        sr - sampling rate
        """
        self.what_is_generated = what_is_generated
        self.sr = sr
        self.how_long = how_long
        self.how_much = how_much
        self.fn = SyntheticDataset._functions if isinstance(fn, str) else fn

    def generate(self, mapping, requirements):
        assert all([x in self.what_is_generated for x in requirements])
        self._get_for_requirements(mapping)
        self._hash = hashlib.sha512(str(np.random.randint(0, 10000000, 10)).encode("utf-8")).digest().hex()[:16]
        # get numpy random state

    def generate_dtype(self, mapping):
        dtype = stage.DType("Array", [self.how_long], np.float32)
        return mapping.output_dtype(dtype)

    @property
    def signature(self):
        return self._hash

    def _get_for_requirements(self, mapping):
        print("Synthetising recordings")
        params = {
            "how_long": self.how_long,
            "sr": self.sr
        }
        n_recs = self.how_much
        dtype = mapping.output_dtype(stage.DType("Array", [self.how_long], np.float32))
        recordings = {}
        for gen in self.what_is_generated:
            recordings[gen] = np.zeros([n_recs] + dtype.shape, dtype.dtype)
        for ix in tqdm.tqdm(range(self.how_much)):
            all_generated = self.fn(**params)
            for gen, data in zip(self.what_is_generated, all_generated):
                data = mapping.map(data)
                key = [ix] + [slice(None, x, None) for x in data.shape]
                recordings[gen].__setitem__(key, data)
        if hasattr(mapping, "normalize"):
            if not mapping.trained:
                print("Applying normalization")
                for gen in self.what_is_generated:
                    recordings[gen] = mapping.normalize(recordings[gen], [dtype.shape[0]] * self.how_much)
        for gen in self.what_is_generated:
            setattr(self, gen, recordings[gen])
            setattr(self, gen + "_lens", [dtype.shape[0]] * self.how_much)


# TODO: this class
class IteratorDataset:
    """
    A dataset class for large or infinite datasets that use generator training interface.
    In progress...
    """
