from os.path import dirname, join

import numpy as np
import pytest

from konrad import atmosphere, utils, RCE


@pytest.fixture
def atmosphere_obj():
    _, phlev = utils.get_pressure_grids(surface_pressure=1000e2, num=50)

    return atmosphere.Atmosphere(phlev=phlev)


class TestRCE:
    def test_init(self, atmosphere_obj):
        """Test initialisation of an RCE simulation."""
        RCE(atmosphere_obj)

    def test_run(self, atmosphere_obj):
        """Run a full RCE simulation and check some outputs."""
        rce = RCE(atmosphere_obj, timestep="12h", max_duration="200d")
        rce.run()

        # Check some basic atmospheric and radiative properties.
        assert 255.5 < rce.radiation["lw_flxu"][-1, -1] < 256.5
        assert rce.atmosphere["T"][-1].min() > 150.0
        assert rce.atmosphere["T"][-1].max() < 288.0
        assert np.all(rce.atmosphere["H2O"] > 0.0)
        assert np.all(rce.atmosphere["H2O"] < 0.015)
