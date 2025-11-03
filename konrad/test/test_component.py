import numpy as np

from konrad.component import Component


class TestComponent:
    def test_to_dataset(self):
        """Test conversion to Xarray dataset."""
        c = Component()

        c.coords = {"dim": np.arange(10)}
        c._data_vars = {"var": (("dim",), np.arange(10))}
        c._attrs = {"title": "Dummy component"}

        ds = c.to_dataset()

        assert ds.title == "Dummy component"
        assert ds["var"][5] == 5

    def test_to_dataset_uninitialized(self):
        """Test conversion to Xarray dataset with uninitialized data."""
        c = Component()

        c.coords = {"dim": np.arange(10)}
        c._data_vars = {"var": (("dim",), None)}

        ds = c.to_dataset()

        assert np.all(np.isnan(ds["var"]))
