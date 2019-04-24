import numpy as np
from konrad.convection import interp_variable


def test_interp_variable():
    a = np.array([-1, 0, 2, 1, 0, 0])
    b = np.array([5, 5, 5, 4, 2, 1])
    assert interp_variable(b, a, 0.5) == 3
