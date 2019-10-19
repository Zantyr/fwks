"""
To do:
- WER
- PER
"""

import dill
import numpy as np
import tqdm

from fwk.acoustic import MappingGenerator


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
    

class WER:
    pass


class PER(Metric):
    requires = ["predicted_phones", "transcriptions"]

    def calculate(self, prediction, transcriptions, **rest):
        print(prediction)
        print(transcripts)
        raise ERRRERADSFGSDFCV


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

    def on_dataset(self, dataset):
        # requirements of metrics should be aggregated
        requirements, data_required = set(), {}
        for metric in self.metrics:
            requirements |= set(metric.requires)
        for req in requirements:
            if req not in ["prediction", "predicted_phones"]:
                data_required[req] = getattr(dataset, req)
                data_required[req + "_lens"] = getattr(dataset, req + "_lens")
        if "prediction" in requirements:
            prediction, prediction_lens = [], []
            for ix in tqdm.tqdm(range(len(dataset.clean_lens))):
                clean = dataset.clean[ix, :dataset.clean_lens[ix]]
                preds = self.model.predict_raw(clean)
                prediction.append(preds)
                prediction_lens.append(len(preds))
            data_required["prediction"] = prediction
            data_required["prediction_lens"] = prediction_lens
        if "predicted_phones" in requirements:
            predicted_phones, predicted_phones_lens = [], []
            for ix in tqdm.tqdm(range(len(dataset.clean_lens))):
                clean = dataset.clean[ix, :dataset.clean_lens[ix]]
                preds = self.model.predict(clean, literal=True)
                predicted_phones.append(preds)
                predicted_phones_lens.append(len(preds))
            data_required["predicted_phones"] = predicted_phones
            data_required["predicted_phones_lens"] = predicted_phones_lens
        # calculate proper predictions
        results = {metric.name: [] for metric in self.metrics}
        for ix in range(len(dataset.clean_lens)):
            # generate data slices according to requirements
            data = {}
            for req in requirements:
                required = data_required[req]
                if isinstance(required, np.ndarray) and hasattr(dataset, req + "_lens"):
                    required_len = getattr(dataset, req + "_lens")
                    required = required[ix, :required_len[ix]]
                else:
                    required = required[ix]
                    if isinstance(required, np.ndarray) and hasattr(dataset, req + "_lens"):
                        required_len = getattr(dataset, req + "_lens")
                        required = required[:required_len[ix]]
                data[req] = required
            for metric in self.metrics:
                results[metric.name].append(metric.calculate(**data))
        return MetricResult(results)
