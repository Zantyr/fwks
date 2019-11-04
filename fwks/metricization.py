"""
To do:
- WER
- PER
"""

import dill
import editdistance
import numpy as np
import random
import tqdm

from fwks.model import MappingGenerator
from fwks.stage import ToDo


class Alphabet:
    """
    Used for Phoneme confusion distance
    """


class Metric:
    requires = ["clean", "dirty"]
    use_test_data = False
    
    @property
    def name(self):
        return self.__class__.__name__
    
    def calculate(self, clean, dirty):
        raise RuntimeError("Plain metric is not callable")


class CosineMetric(Metric):
    def calculate(self, clean, dirty):
        clean = clean.reshape(clean.shape[0], -1)
        dirty = dirty.reshape(dirty.shape[0], -1)
        metrics = (clean * dirty).sum(1) / (
            np.sqrt((np.abs(clean) ** 2 + 2e-12).sum(1)) * \
            np.sqrt((np.abs(dirty) ** 2 + 2e-12).sum(1))
        )
        return metrics.mean()


class EuclidMetric(Metric):
    def calculate(self, clean, dirty):
        clean = clean.reshape(clean.shape[0], -1)
        dirty = dirty.reshape(dirty.shape[0], -1)
        return np.sqrt((np.abs(clean - dirty) ** 2).sum(1)).mean()


class ManhattanMetric(Metric):
    def calculate(self, clean, dirty):
        clean = clean.reshape(clean.shape[0], -1)
        dirty = dirty.reshape(dirty.shape[0], -1)
        return np.sqrt((np.abs(clean - dirty)).sum(1)).mean()
    

class WER(ToDo):
    requires = ["predicted_phones", "transcriptions"]

    def __init__(self, dictionary):
        self.dictionary = dictionary
        

class PER(Metric):
    _requires = ["predicted_phones", "transcriptions"]
    
    def __init__(self, remove_symbols=None, mdl_symbol_map=None, dset_all_phones=None):
        self.remove_symbols = remove_symbols or []
        self.requires = self._requires[:]
        if dset_all_phones is not None:
            self.all_phones = dset_all_phones
        else:
            self.requires.append("dataset:all_phones")
        if mdl_symbol_map is not None:
            self.symbol_map = mdl_symbol_map
        else:
            self.requires.append("model:symbol_map")
            
    def supply(self, k, v):
        setattr(self, k.split(":")[-1], v)
        self.requires.remove(k)
            
    def calculate(self, predicted_phones, transcriptions, **rest):
        # predicted_phones = [x[0] for x in predicted_phones]
        old_predicted_phones = predicted_phones
        predicted_phones = [self.all_phones.index(self.symbol_map[x]) for x in predicted_phones[0] if self.symbol_map[x] not in self.remove_symbols]
        transcriptions = [x for x in transcriptions if self.all_phones[x] not in self.remove_symbols] 
        # 2print(old_predicted_phones, predicted_phones, list(transcriptions))
        dist = editdistance.eval(predicted_phones, transcriptions)
        dist = float(dist) / len(transcriptions)
        return dist


class PhonemeClassConfusion:
    pass


class Metricization:
    def __init__(self, model, compile_mapping=False):
        self.metrics = []
        if isinstance(model, list):
            self.mapping = MappingGenerator(model)
        else:
            self.mapping = model
        self.compile_mapping = compile_mapping
    
    def add_metric(self, metric):
        self.metrics.append(metric)
    
    def on_dataset(self, dataset):
        mapping = self.mapping.get(dataset)
        dataset.generate(mapping, ["clean", "noisy"])
        results = {metric.name: [] for metric in self.metrics}
        for ix in range(len(dataset.clean_lens)):
            clean = dataset.clean[ix, :dataset.clean_lens[ix]]
            noisy = dataset.noisy[ix, :dataset.noisy_lens[ix]]
            for metric in self.metrics:
                results[metric.name].append(metric.calculate(clean, noisy))
        return MetricResult(results)
    
    
class RecordMetricization:
    def __init__(self, model, metrics=None):
        self.metrics = metrics or []
        self.model = model
    
    def add_metric(self, metric):
        self.metrics.append(metric)
    
    def on_dataset(self, dataset):
        mapping = self.model.get_mapping()
        # requirements of metrics should be aggregated
        requirements = set() # ["clean", "noisy"]
        for metric in self.metrics:
            requirements |= set(metric.requires)
        dataset.generate(mapping, list(requirements - set("prediction")))   # should it be regenerated?
        # need to append predictions to the dataset if required
        if "prediction" in requirements:
            prediction, prediction_lens = [], []
            for ix in range(len(dataset.clean_lens)):
                clean = dataset.clean[ix, :dataset.clean_lens[ix]]
                preds = self.model.predict(clean)  # flag for no translation of strings...
                prediction.append(preds)
                prediction_lens.append(len(preds))
        results = {metric.name: [] for metric in self.metrics}
        for ix in range(len(dataset.clean_lens)):
            # generate according to requirements
            data = {}
            for req in requirements:
                required = getattr(dataset, req)
                if isinstance(required, np.ndarray) and hasattr(dataset, req + "_lens"):
                    required_len = getattr(dataset, req + "_lens")
                    required = required[ix, :required_len[ix]]
                else:
                    required = required[ix]
                data[req] = required
            for metric in self.metrics:
                results[metric.name].append(metric.calculate(**data))
        return MetricResult(results)


class MetricResult:
    def __init__(self, results):
        self.results = results
    
    @classmethod
    def load(cls, path):
        res = dill.load(path)
        return MetricResult(res)
    
    def save(self, path):
        dill.dump(path, self.results)
    
    def summary(self):
        for k, v in self.results.items():
            s = "{}: {} +- {}".format(k, np.array(v).mean(), np.array(v).std())
            print(s)

            
            
class MetricizationAB:
    def __init__(self, metrics=None):
        self.metrics = metrics or []
        self.results = {}
    
    def add_metric(self, metric):
        self.metrics.append(metric)
    
    def calculate(self, clean, clean_lens, noisy, noisy_lens):
        self.results = {}
        for metric in self.metrics:
            self.results[metric.name] = []
        for clean_rec, clean_len, noisy_rec, noisy_len in tqdm.tqdm(zip(clean, clean_lens, noisy, noisy_lens), total=len(clean_lens)):
            clean_rec = clean_rec[:clean_len]
            noisy_rec = noisy_rec[:noisy_len]
            for metric in self.metrics:
                self.results[metric.name].append(metric.calculate(clean_rec, noisy_rec))

    def summary(self):
        for k, v in self.results.items():
            print("{}: {} +- {}".format(k, np.array(v).mean(), np.array(v).std()))
            

class TrainedModelMetricization:
    def __init__(self, model, metrics=None):
        self.model = model
        self.metrics = metrics or []
        self.results = {}
    
    def add_metric(self, metric):
        self.metrics.append(metric)

    def _get_requirements(self, dataset):
        # requirements of metrics should be aggregated
        requirements, data_required = set(), {}
        for metric in self.metrics:
            requirements |= set(metric.requires)
        requirements = list(requirements)
        for req in requirements[:]:
            # print(req)
            if ":" in req:
                # print("removing")
                if req.split(":")[0] == "dataset":
                    value = getattr(dataset, req.split(":")[1])
                elif req.split(":")[0] == "model":
                    value = getattr(self.model, req.split(":")[1])
                else:
                    raise AttributeError("Strange requirement to metric: " + req)
                for metric in self.metrics:
                    if req in metric.requires:
                        metric.supply(req, value)
                requirements.remove(req)
            elif req not in ["prediction", "predicted_phones"]:
                data_required[req] = getattr(dataset, req)
                data_required[req + "_lens"] = getattr(dataset, req + "_lens")
        return requirements, data_required        
        
    def on_dataset(self, dataset, partial=None):
        requirements, data_required = self._get_requirements(dataset)
        selectables = list(data_required.keys())
        # print(requirements, data_required)
        if partial is None:
            selection = lambda: range(len(dataset.clean_lens))
        else:
            subselection = random.sample(range(len(dataset.clean_lens)), partial)
            selection = lambda: subselection
        if "prediction" in requirements:
            prediction, prediction_lens = [], []
            for ix in tqdm.tqdm(selection()):
                clean = dataset.clean[ix, :dataset.clean_lens[ix]]
                preds = self.model.predict_raw(clean, use_mapping=False)
                prediction.append(preds)
                prediction_lens.append(len(preds))
            data_required["prediction"] = prediction
            data_required["prediction_lens"] = prediction_lens
        if "predicted_phones" in requirements:
            predicted_phones, predicted_phones_lens = [], []
            for ix in tqdm.tqdm(selection()):
                clean = dataset.clean[ix, :dataset.clean_lens[ix]]
                preds = self.model.predict(clean, literal=False, use_mapping=False)
                predicted_phones.append(preds)
                predicted_phones_lens.append(len(preds))
            data_required["predicted_phones"] = predicted_phones
            data_required["predicted_phones_lens"] = predicted_phones_lens
        # calculate proper predictions
        results = {metric.name: [] for metric in self.metrics}
        print("Calculating metrics...")
        for ix, sel in tqdm.tqdm(enumerate(selection())):
            # generate data slices according to requirements
            data = {}
            for req in requirements:
                selector = sel if req in selectables else ix
                # print(ix, req, selector)
                required = data_required[req]
                if isinstance(required, np.ndarray) and hasattr(dataset, req + "_lens"):
                    required_len = getattr(dataset, req + "_lens")
                    required = required[selector, :required_len[selector]]
                else:
                    required = required[selector]
                    if isinstance(required, np.ndarray) and hasattr(dataset, req + "_lens"):
                        required_len = getattr(dataset, req + "_lens")
                        required = required[:required_len[selector]]
                data[req] = required
            for metric in self.metrics:
                results[metric.name].append(metric.calculate(**data))
        return MetricResult(results)
