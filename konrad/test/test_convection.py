import konrad
from konrad.convection import energy_threshold


def test_energy_threshold():
    assert energy_threshold(konrad.surface.SurfaceFixedTemperature()) == 10**-8
    surface = konrad.surface.SurfaceHeatCapacity()
    assert energy_threshold(surface) == float(surface.heat_capacity / 1e13)
