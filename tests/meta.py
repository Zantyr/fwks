from fwks.dataset import Dataset

def get_test_dataset():
    dset = Dataset()
    dset.loader_adapter = "plain"
    dset.get_from("tests/test_data")
    return dset
