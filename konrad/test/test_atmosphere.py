from os.path import (dirname, join)

import numpy as np
import pytest

from konrad import (atmosphere, utils)


@pytest.fixture
def atmosphere_obj():
    _, phlev = utils.get_pressure_grids(surface_pressure=1000e2, num=50)

    return atmosphere.Atmosphere(phlev=phlev)


class TestAtmosphere:
    ref_dir = join(dirname(__file__), "reference_data", "")

    def test_init(self):
        """Test basic initialization of the atmosphere component."""
        phlev = np.array([1000e2, 750e2, 500e2, 100e2, 10e2, 1e2])

        atmosphere.Atmosphere(phlev=phlev)

    def test_cold_point_index(self, atmosphere_obj):
        """Test retrieval of the cold point index."""
        assert atmosphere_obj.get_cold_point_index() == 11

    def test_triple_point_index(self, atmosphere_obj):
        """Test retrieval of the triple point index."""
        assert atmosphere_obj.get_triple_point_index() == 2

    def test_cold_point_plev(self, atmosphere_obj):
        """Test retrieval of the cold point pressure."""
        assert np.isclose(atmosphere_obj.get_cold_point_plev(), 19611.012)

    def test_triple_point_plev(self, atmosphere_obj):
        """Test retrieval of the triple point plev."""
        assert np.isclose(atmosphere_obj.get_triple_point_plev(), 73875.426)

    def test_from_netcdf(self):
        """Test initialisation from netCDF file."""
        ncfile = join(self.ref_dir, 'reference.nc')
        atmosphere.Atmosphere.from_netcdf(ncfile)

    def test_from_netcdf_invalid_input(self):
        """Check exception when passing unused keywords."""
        # Old versions of the ``Atmosphere`` API allowed to pass unused keyword
        # arguments to the ``from_*`` classmethods. This lead to unexpected
        # behaviour during initialisation.
        with pytest.raises(TypeError):
            atmosphere.Atmosphere.from_netcdf('dummy.nc', surface=None)

    def test_refine_plev(self, atmosphere_obj):
        """Test refinement of pressure grid."""
        phlev = np.array([1000e2, 500e2, 10e2])
        atmosphere_new = atmosphere_obj.refine_plev(phlev=phlev)

        #TODO Add a proper test of values

