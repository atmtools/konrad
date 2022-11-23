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
        """Integrate an RCE simulation for four time steps.."""
        rce = RCE(atmosphere_obj, timestep="12h", max_duration="48h")
        rce.run()
