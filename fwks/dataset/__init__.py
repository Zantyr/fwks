"""
fwks.dataset
============

Contains implementation of datasets and methods to prepare data for later processing. Includes
classes meant to find the list of files and load them, forming all required data sources.

Dataset contains data sources. Data source may be a set of recordings, metadata and so on, indexed by integers
(as a list or ndarray). Dataset uses a set of adapters to extend the capabilities by loading from diverse paths,
preparing other versions of data beyond simple set of recordings etc. ``fwks.dataset.adapters.LoaderAdapter`` loads instances from
specific path and supplies basic loader functions and ``fwks.dataset.adapters.MapperAdapter`` extends the dataset by creating derivatives of the data.
"""

__all__ = ["PlainAdapter", "ClarinAdapter", "MixturesMapper",
           "Dataset", "SyntheticDataset", "IteratorDataset"]

from .adapters import PlainAdapter, ClarinAdapter, MixturesMapper
from .dataset import Dataset, SyntheticDataset, IteratorDataset
