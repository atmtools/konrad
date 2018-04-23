# -*- coding: utf-8 -*-
"""This module contains classes handling different treatments of ozone."""


import abc
import logging

import numpy as np
from scipy.interpolate import interp1d

__all__ = [
    'Ozone',
    'Ozone_pressure',
    'Ozone_height',
]

logger = logging.getLogger(__name__)

class Ozone(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for ozone treatments."""

    @abc.abstractmethod
    def get(self, o3, z, z_new, p, cp, cp_new):
        """Return the new ozone profile.

        Parameters:
            o3 (ndarray): old ozone profile
            z (ndarray): old height profile
            z_new (ndarray): new height profile

        Returns:
            ndarray: new ozone profile
        """


class Ozone_pressure(Ozone):
    """Ozone fixed with pressure, no adjustment needed."""
    def get(self, o3, *args, **kwargs):
        return o3


class Ozone_height(Ozone):
    """Ozone fixed with height."""
    def get(self, o3, z, z_new, *args, **kwargs):
        f = interp1d(z, o3, fill_value='extrapolate')
        return f(z_new)


class Ozone_normed_pressure(Ozone):
    """Ozone shifts with cold point."""
    def get(self, o3, z, z_new, p, cp, cp_new):
        f = interp1d(np.log(p/cp), o3, fill_value='extrapolate')
        return f(np.log(p/cp_new))

