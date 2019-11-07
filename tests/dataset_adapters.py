import unittest

import fwks.dataset

adapter = MixAdapter(n_channels=2)  # should mix two recordings and generate 

model = fwks.model.from_library("denoising", channels=2) # library should store predefined models, configurable

dset = fwks.dataset.Dataset()
dset.get_from("")
dset.loader_adapter = "plain"
dset.add_mapping_adapter(adapter)
model.build(dset) # should generate ["mixtures", "channel_1", "channel_2"]

metric = SeparationMetricization()
metric.add_metric(PermutativeMetric(SNR()))

# metricize?
