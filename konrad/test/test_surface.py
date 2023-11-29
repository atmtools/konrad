import numpy as np

from konrad import surface


class TestSurface:
    def test_init(self):
        """Test basic initialization of the surface component."""
        init_args = (
            280,
            280.0,
            np.array(280),
            np.array(280.0),
            np.array([280]),
            np.array([280.0]),
            [280],
            [280.0],
        )

        for t in init_args:
            surf = surface.FixedTemperature(temperature=t)
            assert np.array_equal(surf["temperature"], np.array([280.0], dtype=float))
