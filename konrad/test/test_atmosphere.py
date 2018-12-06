import numpy as np

from konrad import atmosphere


class TestAtmosphere:
    def test_init(self):
        """Test basic initialization of the atmosphere component."""
        plev = np.array([1000e2, 750e2, 500e2, 100e2, 10e2, 1e2])

        atmosphere.Atmosphere(plev=plev)
