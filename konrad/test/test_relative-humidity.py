from os.path import (dirname, join)

import numpy as np
import pytest

from konrad import (atmosphere, utils, humidity)


@pytest.fixture
def atmosphere_obj():
    _, phlev = utils.get_pressure_grids(surface_pressure=1000e2, num=50)

    return atmosphere.Atmosphere(phlev=phlev)


class TestRelativeHumidity:
    rh_models = (
        humidity.CacheFromAtmosphere,
        humidity.HeightConstant,
        humidity.VerticallyUniform,
        humidity.ConstantFreezingLevel,
        humidity.FixedUTH,
        humidity.CoupledUTH,
        humidity.CshapeConstant,
        humidity.CshapeDecrease,
        humidity.Manabe67,
        humidity.Cess76,
        humidity.Romps14,
    )

    @pytest.mark.parametrize("rh_model", rh_models)
    def test_cold_point_index(self, atmosphere_obj, rh_model):
        """Test initialisation and call of all relative humidity models."""
        rh_model()(
            atmosphere=atmosphere_obj,
            convection={"convective_top_plev": [100e2]},
            surface={"temperature": [300.0]},
        )
