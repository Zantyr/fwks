import abc
import os
import scipy.io.wavfile as sio
import tqdm


class Adapter(metaclass=abc.ABCMeta):
    @property
    def loader_function(self):
        """
        If overridden, should accept filename and return (sample_rate, data)
        """
        return sio.read

    def get_recording_lengths(self):
        lens = []
        print("Getting dataset lengths")
        for f, _ in tqdm.tqdm(zip(*self.get_fnames())):
            sr, data = sio.read(f)
            assert sr == 16000
            assert len(data.shape) == 1
            lens.append(len(data))
        return lens

    @abc.abstractmethod
    def get_fnames(self):
        return rec_fnames, trans_fnames


class PlainAdapter(Adapter):
    def __init__(self, root):
        self.ROOT = root

    def get_fnames(self):
        rec_fnames, trans_fnames = [], []
        for i in os.listdir(self.ROOT):
            if i.endswith(".wav"):
                rec_fnames.append(os.path.join(self.ROOT, i))
                txt = os.path.join(self.ROOT, i.replace(".wav", ".txt"))
                if txt not in trans_fnames:
                    trans_fnames.append(txt)
        return rec_fnames, trans_fnames


class ClarinAdapter(Adapter):
    def __init__(self, root):
        self.ROOT = root

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
