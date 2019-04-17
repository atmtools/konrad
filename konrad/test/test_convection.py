import numpy as np
import konrad
from konrad.convection import energy_threshold, interp_variable


def test_energy_threshold():
    assert energy_threshold(konrad.surface.SurfaceFixedTemperature()) == 10**-8
    surface = konrad.surface.SurfaceHeatCapacity()
    assert energy_threshold(surface) == float(surface.heat_capacity / 1e13)


def test_interp_variable():
    a = np.array([-1, 0, 2, 1, 0, 0])
    b = np.array([5, 5, 5, 4, 2, 1])
    assert interp_variable(b, a, 0.5) == 3
