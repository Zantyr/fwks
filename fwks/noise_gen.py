import numpy as np
import os
import random
import scipy.io.wavfile as sio
import subprocess
import tempfile


class NoiseGenerator:
    def _pre(self, data):
        return data

    def _post(self, data):
        return data
        
    def pre(self, data):
        shape, dtype = data.shape, data.dtype
        data = self._pre(data)
        assert shape == data.shape
        assert dtype == data.dtype
        return data
        
    def post(self, data):
        shape, dtype = data.shape, data.dtype
        data = self._post(data)
        assert shape == data.shape
        assert dtype == data.dtype
        return data


class Static(NoiseGenerator):
    def __init__(self, power=0.1):
        self.power = power
        
    def _pre(self, data):
        data = data + np.random.normal(size=data.shape).astype(np.float32) * self.power
        return data

    
class RandomImpulseResponse(NoiseGenerator):
    def __init__(self, dataset_path):
        self.ir = []
        for fname in os.listdir(dataset_path):
            if fname.endswith("wav"):
                sr, data = sio.read(os.path.join(dataset_path, fname))
                assert sr == 16000
                self.ir.append(data)

    def _pre(self, data):
        impulse_resp = random.choice(self.ir)[:len(data)]
        return np.convolve(data, impulse_resp, 'same')

    
class FileNoise(NoiseGenerator):
    def __init__(self, dataset_path, snr):
        pass
    

class CodecSox(NoiseGenerator):
    
    _formats = {
        "gsm": {"ext": ".gsm", "opt": []},
        "amr-nb-lq": {"ext": ".amr-nb", "opt": ["-C", "0"]},
        "amr-nb-hq": {"ext": ".amr-nb", "opt": ["-C", "7"]},
        "amr-wb": {"ext": ".amr-wb", "opt": []},
        "mp3-lq": {"ext": ".mp3", "opt": ["-C", "96.10"]},
        "mp3-hq": {"ext": ".mp3", "opt": ["-C", "192.1"]},
    }
    
    def __init__(self, fmt):
        rc = subprocess.Popen(["which", "sox"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()
        assert rc == 0
        self.format = self._formats[fmt]
        
    def _pre(self, data):
        oldname = tempfile.mktemp() + ".wav"
        tmpname = tempfile.mktemp() + self.format['ext']
        newname = tempfile.mktemp() + ".wav"
        sio.write(oldname, 16000, data)
        subprocess.Popen(['sox', oldname] + self.format['opt'] + [tmpname]).communicate()
        subprocess.Popen(['sox', tmpname, '-r', '16000', '-e', 'signed', '-b', '16', newname]).communicate()
        newdata = sio.read(newname)[1]
        os.remove(oldname)
        os.remove(tmpname)
        os.remove(newname)
        data = np.zeros_like(data)
        data[:min(len(data), len(newdata))] = newdata[:min(len(data), len(newdata))]
        return data
    
