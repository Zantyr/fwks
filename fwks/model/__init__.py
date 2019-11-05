"""
model
============
Library for all types of models. Model is a single, serializable instance of a pipeline that fulfills particular
processing job, either from preprocessed or raw data, mostly the latter.

Models define the mode of training.
"""

__all__ = [
    "AcousticModel",
    "MappingGenerator",
    "ItemLoader",
    "DenoisingModel"
]

from .acoustic import AcousticModel, MappingGenerator
from .meta import ItemLoader
from .denoising import DenoisingModel


# from .language import LanguageModel  # check capabilities
