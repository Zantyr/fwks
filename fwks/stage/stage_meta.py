import abc

import keras
from syntax import Show


class Watcher(type):
    """
    Based on https://stackoverflow.com/questions/18126552
    """

    count = 0
    
    def __init__(cls, name, bases, clsdict):
        if len(cls.mro()) > 2:
            Watcher.count += 1
            print("Yet another class to be finished: " + name)
        super(Watcher, cls).__init__(name, bases, clsdict)


class ToDo(metaclass=Watcher):
    """
    This class will print that something is to be done
    """
    
    @staticmethod
    def status():
        print("Classes to be finished: {}".format(Watcher.count))


class SelectionAdapter(metaclass=abc.ABCMeta):
    """
    SelectionAdapter is a class for division between train, valid and test datasets
    The division may or may not take into account the speakers or other circumstances
    """
    def initialize(self, dataset):
        number = len(dataset.rec_fnames)
        train_threshold = 1. - self._valid_percentage - self._test_percentage
        valid_threshold = 1. - self._test_percentage
        selection = np.random.random(number)
        self._train = selection < train_threshold
        self._valid = (train_threshold < selection) & (selection <= valid_threshold)
        self._test = selection >= valid_threshold
        self._hash = hashlib.sha512(selection.tobytes()).digest().hex()[:16]

    @property
    def selection_hash(self):
        return self._hash
    
    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test
    
    @abc.abstractmethod
    def serialize(self):
        pass


class Stage(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def trainable(self):
        pass

    @abc.abstractmethod
    def output_dtype(self, input_dtype):
        pass

    def bind(self, previous):
        self.previous = previous
        return self
    
    @abc.abstractmethod
    def map(self, recording):
        pass

    
class NetworkableMixin(metaclass=abc.ABCMeta):
    """
    You can add this Analytic to network
    """


class Loss(Stage, metaclass=abc.ABCMeta):
    """
    Loss is a strange Stage, as it does not really implement most of the methods
    TODO: To rework
    """    
    
    @property
    def trainable(self):
        return False

    def output_dtype(self, input_dtype):
        return None

    def bind(self, previous):
        return None
    
    def map(self, recording):
        return None

    @property
    @abc.abstractmethod    
    def selection_hash(self):
        pass
    
    @property
    @abc.abstractmethod
    def requirements(self):
        if self.use_noisy:
            return ["noisy", "transcripts"]
        return ["clean", "transcripts"]

    @abc.abstractmethod
    def fetch_train(self, dataset):
        pass
        
    @abc.abstractmethod
    def fetch_valid(self, dataset):
        pass
    
    @abc.abstractmethod
    def fetch_test(self, dataset):
        pass

    @abc.abstractmethod
    def serialize(self):
        pass


class DType(Show):
    def __init__(self, cls, shape, dtype):
        self.cls = cls
        self.shape = shape
        self.dtype = dtype
        
    @classmethod
    def of(cls, what):
        return NotImplementedError("")
    
    
class Neural(Stage):
    
    """
    Make all neural stages compose with each other...
    """
    
    def __init__(self):
        super().__init__()
    
    @property
    def trainable(self):
        return True
    
    def output_dtype(self, input_dtype):
        raise RuntimeError("Not used")
        
    def map(self, recording):
        """
        Stages should have different interfaces, this method is not required
        """
        raise RuntimeError("Not used")
    
    def new_network(self, dtype):
        """
        Should create a new network giving an Input layer as an input to the model
        """
        input_layer = keras.layers.Input([None] + list(dtype.shape[1:]))
        return keras.models.Model(input_layer, self.get_graph()(input_layer))

    def join(self, previous):
        """
        When joining, should get the output of the previous model and pass
        it as an input to the new network
        """
        return keras.models.Model(previous.inputs, self.get_graph()(previous.outputs[0]))


class CustomNeural(Neural):    
    """
    Neural models, which have custom implementation.
    """
    def __init__(self, graph):
        super().__init__()
        self._graph = graph
        
    def get_graph(self):
        return self._graph



class Analytic(Stage):
    @property
    def trainable(self):
        False

    def map(self, recording):
        if self.previous is None:
            return self._function(recording)
        return self._function(self.previous.map(recording))

    def serialize(self):
        return self

    
class Normalizer(Stage):
    @property
    def trainable(self):
        False

    def output_dtype(self, input_dtype):
        if self.previous:
            input_dtype = self.previous.output_dtype(input_dtype)
        return input_dtype

    def serialize(self):
        return self
    
    @abc.abstractmethod
    def normalize(self, features, lengths):
        pass