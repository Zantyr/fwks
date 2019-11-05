"""
fwks.dataset
============

Contains implementation of datasets and methods to prepare data for later processing. Includes
classes meant to find the list of files and load them, forming all required data sources.

Dataset contains data sources. Data source may be a set of recordings, metadata and so on, indexed by integers
(as a list or ndarray). Dataset uses a set of adapters to extend the capabilities by loading from diverse paths,
preparing other versions of data beyond simple set of recordings etc. LoaderAdapter loads instances from
specific path and MapperAdapter extends the dataset by creating derivatives of the data.
"""

__all__ = ["Dataset", "SyntheticDataset"]

from .dataset import Dataset, SyntheticDataset
