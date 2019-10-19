import dill
import keras
import numpy as np
import os
import shutil
import tempfile
import time
import zipfile

import keras.backend as K
import tensorflow as tf

from fwks.stage import Neural, Analytic, Loss, DType, Stage
from fwks.miscellanea import PeekGradients, StopOnConvergence

_defaults = {"tf": tf}    # this should contain everything...


class Representation:
    def __init__(self, mapping, model):
        self.mapping = mapping
        self.model = model

    def map(self, recording):
        mapped = self.mapping.map(recording)
        if self.model:
            mapped = np.stack([mapped])
            mapped = self.model.predict(mapped)[0]
        return mapped


class MappingGenerator:
    def __init__(self, stages, representation_counter=None):
        self.stages = stages
        self.representation_mapping = None
        self.representation_counter = representation_counter

    def get_all(self, dataset):
        representation_cnt = 0
        mapping, train_network, network, loss = None, None, None, None
        stages = list(reversed(self.stages))
        while stages:
            if isinstance(stages[-1], Neural) or isinstance(stages[-1], Loss):
                break
            mapping = stages.pop().bind(mapping)
            representation_cnt += 1
        dset_dtype = dataset.generate_dtype(mapping)
        stages.reverse()
        for stage in stages:
            if isinstance(stage, Neural):
                if network is None:
                    network = stage.new_network(mapping.output_dtype(dset_dtype))
                else:
                    network = stage.join(network)
            elif isinstance(stage, Loss):
                train_network = stage.compile(network)
                loss = stage
                break
            elif isinstance(stage, Analytic):
                if hasattr(stage, "to_network"):
                    network = stage.to_network(network)
                else:
                    raise TypeError("Analytic joint to network without to_network() method")
            else:
                raise TypeError("Incorrect subtype of Stage")
            representation_cnt += 1
            if representation_cnt == self.representation_counter:
                self.representation_mapping = keras.models.Model(network.inputs, network.outputs)
        else:
            if network is not None:
                raise RuntimeError("Network has not been compiled")
        return mapping, train_network, network, loss

    def get(self, dataset):
        return self.get_all(dataset)[0]



class ItemLoader:
    """
    Creates loadable item from object
    Any pickleability should be done here
    """

    def __init__(self, item):
        self.builder = None
        if isinstance(item, Stage):
            if hasattr(item, "serialize"):
                self.value = item.serialize()
                if isinstance(self.value, tuple):
                    self.builder, self.value = self.value
            else:
                self.value = item.__class__
        elif any([isinstance(item, x) for x in (keras.callbacks.Callback,)]):
            self.value = item.__class__
        else:
            raise RuntimeError("No loader for {}".format(item))

    def load(self):
        if self.builder:
            return self.builder(self.value)
        return self.value


class SoundModel:

    _separate_lists = ["callbacks", "stages"]


    def __init__(self, stages, name=None, symbol_map=None, callbacks=None, num_epochs=250, checkpoint_path=None):
        self.stages = stages
        self.symbol_map = symbol_map
        self.dataset_signature = None
        self.split_signature = None
        self.built = False
        self.config = None
        self.name = name if name else "blind"
        self.metrics = []
        self.statistics = {}
        self.complexity, self.building_time = None, None
        self.callbacks = callbacks or [
            keras.callbacks.TerminateOnNaN(),
            StopOnConvergence(4)
        ]
        self.representation_mapping = None
        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path

    def _save_dill_builder(self, item, path):
        item = self._summarize(item, path)
        print(path)
        with open(path, "wb") as f:
            new_one = dill.dump(item, f)

    def _summarize(self, item, path):
        if any([isinstance(item, x) for x in (tuple, list, set)]):
            return [self._summarize(x, path) for x in item]
        elif any([isinstance(item, x) for x in (dict, )]):
            return {k: self._summarize(v, path) for k, v in item.items()}
        elif any([isinstance(item, x) for x in (str, int, float, type(None))]):
            return item
        else:
            return ItemLoader(item)

    @classmethod
    def _unsummarize(self, item, tmpdname):
        if isinstance(item, ItemLoader):
            return item.load()
        elif hasattr(item, "__iter__"):
            if hasattr(item, "__items__"):
                return {k: self._unsummarize(v, tmpdname) for k, v in item.items()}
            else:
                return [self._unsummarize(k, tmpdname) for k in item]
        else:
            return item

    def get_metrics(self, show=False):
        if show:
            return print(self.statistics.items())
        return self.statistics.items()

    def get_mapping(self):
        return Representation(self.mapping, self.representation_mapping)

    def predict_raw(self, recording):
        if len(recording.shape) < 2:
            recording = np.stack([recording])
        dtype = self.mapping.output_dtype(DType("Array", recording.shape[1:], recording.dtype))
        mapped = np.zeros([recording.shape[0]] + dtype.shape, dtype=dtype.dtype)
        for i in range(recording.shape[0]):
            mapped[i] = self.mapping.map(recording[i])
        return self.network.predict(mapped)

    def save(self, path, format=False, save_full=True):
        if format:
            pass # TODO: change path somehow
        tmpdname = tempfile.mkdtemp()
        try:
            separate_objects = {}
            for k in dir(self):
                v = getattr(self, k)
                if isinstance(v, keras.models.Model):
                    fname = k + ".h5"
                    v.save(os.path.join(tmpdname, fname))
                    node_path = "keras://" + fname
                    separate_objects[k] = (node_path, v)
                elif k in self._separate_lists:
                    fname = k + ".bin"
                    self._save_dill_builder(v, os.path.join(tmpdname, fname))
                    node_path = "dill-builder://" + fname
                    separate_objects[k] = (node_path, v)
            try:
                for k, v in separate_objects.items():
                    setattr(self, k, v[0])
                with open(os.path.join(tmpdname, "root.dill"), "wb") as f:
                    dill.dump(self, f)
                with zipfile.ZipFile(path, 'w') as zipf:
                    for folder, _, files in os.walk(tmpdname):
                        for fname in files:
                            zipf.write(os.path.join(folder, fname), fname)
            finally:
                for k, v in separate_objects.items():
                    setattr(self, k, v[1])
        finally:
            shutil.rmtree(tmpdname)

    @classmethod
    def load(self, path):
        tmpdname = tempfile.mkdtemp()
        print(os.listdir(tmpdname))
        try:
            with zipfile.ZipFile(path) as f:
                f.extractall(tmpdname)
            with open(os.path.join(tmpdname, "root.dill"), "rb") as f:
                new_one = dill.load(f)
            for k, v in new_one.__dict__.items():
                if isinstance(v, str):
                    if v.startswith("dill://"):
                        fname = os.path.join(tmpdname, v.split("://")[1])
                        with open(fname, "rb") as f:
                            new_value = dill.load(f)
                        setattr(new_one, k, new_value)
                    elif v.startswith("keras://"):
                        fname = os.path.join(tmpdname, v.split("://")[1])
                        new_value = keras.models.load_model(fname, custom_objects=_defaults)
                        # TODO: Add decompression of additional features (like custom tf functions)
                        setattr(new_one, k, new_value)
                    elif v.startswith("dill-builder://"):
                        fname = os.path.join(tmpdname, v.split("://")[1])
                        with open(fname, "rb") as f:
                            new_value = dill.load(f)
                        new_value = self._unsummarize(new_value, tmpdname)
                        print(new_value)
                        setattr(new_one, k, new_value)
            return new_one
        finally:
            shutil.rmtree(tmpdname)

    def build(self, dataset, **config):
        start_time = time.time()
        self.config = config
        mapping_generator = MappingGenerator(self.stages)
        mapping, train_network, network, loss = mapping_generator.get_all(dataset)
        dataset.generate(mapping, loss.requirements)
        self.dataset_signature = dataset.signature
        if network is not None:
            callbacks = self.callbacks[:]
            if self.checkpoint_path is not None:
                mc = keras.callbacks.ModelCheckpoint('{}{}{}-{}.h5'.format(self.checkpoint_path,
                    os.path.sep, self.name, "{epoch:04d}"),
                             save_weights_only=False, period=5)
                callbacks.append(mc)
            config = {
                "batch_size": 32,
                "callbacks": callbacks,
                "validation_data": loss.fetch_valid(dataset),
                "epochs": self.num_epochs,
            }
            self.network = network
            train_network.summary()
            train_network.fit(*loss.fetch_train(dataset), **config)
        self.statistics["loss"] = train_network.evaluate(*loss.fetch_test(dataset))
        self.split_signature = loss.selection_hash
        for metric in self.metrics:
            self.statistics[metric.name] = metric.calculate(*loss.fetch_test(dataset))
        # predict and calculate - loss has to have "calculate"
        self.building_time = time.time() - start_time
        self.complexity = None
        self.mapping = mapping
        if hasattr(dataset, "all_phones"):
            self.symbol_map = dataset.all_phones
        self.built = True

    def add_metric(self, metric):
        self.metrics.append(metric)

