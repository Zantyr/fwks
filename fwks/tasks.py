import os
import fwks.model as model
import fwks.dataset as dataset
import fwks.metricization as metricization

"""
TODO:
- saving // loading
- running the network
- creation of chains for language models
- test coverage
"""


class Task(type):
    
    _instances = {}
    
    @classmethod
    def all(cls):
        return [cls._instances[x] for x in sorted(cls._instances.keys())]
    
    def __new__(self, name, bases, dct):
        new_dct = {"name": name, "implemented": True}
        new_dct.update(dct)
        item = super().__new__(self, name, bases, new_dct)
        self._instances[name] = item
        return item
    
    
def make_training_task(
        noise=None,
        evaluation_metrics=None
    ):
    # TODO: add training using noisy instead of clean

    
    class AbstractModelTraining(Task):
    
        how_much = 9000
        noise_gen = noise
        epochs = 250
        from_path = "datasets/clarin-long/data"
        metrics = evaluation_metrics or []

        def __new__(self, name, bases, dct):
            this = self
            _metrics = self.metrics
            
            @classmethod
            def get_dataset(self):
                dset = dataset.Dataset(noise_gen=this.noise_gen)
                dset.loader_adapter = "clarin"
                dset.get_from(self.from_path)
                return dset

            @classmethod
            def validate(self, cache):
                return os.path.exists(os.path.join(cache, "model.zip"))

            @classmethod
            def run(self, cache):
                try:
                    if not os.path.exists(cache):
                        os.mkdir(cache)
                except:
                    pass
                dset = self.get_dataset()
                dset.select_first(self.how_much)
                am = self.get_acoustic_model()
                am.num_epochs = this.epochs
                am.name = name
                am.build(dset)
                am.summary()
                if self._metrics:
                    metric_obj = metricization.TrainedModelMetricization(am, self._metrics)
                    results = metric_obj.on_dataset(dset)
                    results.summary()
                am.save(os.path.join(cache, "model.zip"), save_full=True)
                print("=" * 60)
                print("Task done!\n")

            @classmethod
            def summary(self, cache, show=False):
                try:
                    print(cache)
                    am = model.AcousticModel.load(os.path.join(cache, "model.zip"))
                    return am.summary(show=show)
                except FileNotFoundError:
                    print("Cannot find the model archive - aborting")

            new_dct = {"run": run, "validate": validate, "summary": summary, 
                       "how_much": this.how_much, "get_dataset": get_dataset,
                       "_metrics": _metrics}
            new_dct.update(dct)
            return super().__new__(self, name, bases, new_dct)
        
        @classmethod
        def add_metric(self, metric):
            self.metrics.append(metric)

    metaclass = AbstractModelTraining    
    return metaclass

AbstractModelTraining = make_training_task()


def make_ab_feature_test(noise_gen):
    _noise_gen = noise_gen
    
    class AbstractABTraining(Task):
    
        how_much = 9000
        noise_gen = _noise_gen
        from_path = "datasets/clarin-long/data"

        def __new__(self, name, bases, dct):
            this = self

            @classmethod
            def get_dataset(self):
                dset = dataset.Dataset(noise_gen=this.noise_gen)
                dset.loader_adapter = "clarin"
                dset.get_from(self.from_path)
                return dset

            @classmethod
            def validate(self, cache):
                pass

            @classmethod
            def run(self, cache):
                try:
                    if not os.path.exists(cache):
                        os.mkdir(cache)
                except:
                    pass
                dset = self.get_dataset()
                dset.select_first(self.how_much)
                am = self.get_acoustic_model()
                mapping_generator = model.MappingGenerator(am.stages)             
                mapping = mapping_generator.get(dset)
                dset.generate(mapping, ["clean", "noisy"])
                print("Shape of the data: {}".format(dset.clean.shape))
                metric_obj = metricization.MetricizationAB([
                    metricization.CosineMetric(),
                    metricization.EuclidMetric(),
                    metricization.ManhattanMetric()
                ])
                diff = (dset.clean - dset.noisy)
                metric_obj.calculate(
                    dset.clean, dset.clean_lens,
                    dset.noisy, dset.noisy_lens
                )
                metric_obj.summary()
                print("=" * 60)
                print("Task done!\n")

            @classmethod
            def summary(self, cache, show=False):
                pass

            new_dct = {"run": run, "validate": validate, "summary": summary, 
                       "how_much": this.how_much, "get_dataset": get_dataset}
            new_dct.update(dct)
            return super().__new__(self, name, bases, new_dct)

    return AbstractABTraining
