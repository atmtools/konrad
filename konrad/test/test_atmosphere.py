import numpy as np
import pytest

from konrad import (atmosphere, utils)


@pytest.fixture
def atmosphere_obj():
    plev, _ = utils.get_pressure_grids(1000e2, 1, 50)

    return atmosphere.Atmosphere(plev=plev)


class TestAtmosphere:
    def test_init(self):
        """Test basic initialization of the atmosphere component."""
        plev = np.array([1000e2, 750e2, 500e2, 100e2, 10e2, 1e2])

        atmosphere.Atmosphere(plev=plev)

    def test_cold_point_index(self, atmosphere_obj):
        """Test retrieval of the cold point index."""
        assert atmosphere_obj.get_cold_point_index() == 11

    def test_triple_point_index(self, atmosphere_obj):
        """Test retrieval of the triple point index."""
        assert atmosphere_obj.get_triple_point_index() == 2

    def test_cold_point_plev(self, atmosphere_obj):
        """Test retrieval of the cold point pressure."""
        assert np.isclose(atmosphere_obj.get_cold_point_plev(), 20023.067)

    def test_triple_point_plev(self, atmosphere_obj):
        """Test retrieval of the triple point plev."""
        assert np.isclose(atmosphere_obj.get_triple_point_plev(), 74287.948)
