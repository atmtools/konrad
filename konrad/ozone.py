# -*- coding: utf-8 -*-
"""This module contains classes handling different treatments of ozone."""

import abc
import logging
from scipy.interpolate import interp1d

__all__ = [
    'Ozone',
    'Ozone_pressure',
    'Ozone_height',
    'Ozone_normed_pressure',
]

logger = logging.getLogger(__name__)

class Ozone(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for ozone treatments."""

    def __init__(self, initial_ozone=None, initial_height=None,
                 norm_level=None):

        #TO DO: include a default initial ozone profile
        self.initial_ozone = initial_ozone

        self.initial_height = initial_height
        self.norm_level = norm_level

    @abc.abstractmethod
    def get(self, height_new, p, norm_new, **kwargs):
        """Return the new ozone profile.
        Parameters:
            height_new (ndarray): height array of the current atmosphere
                (for Ozone_height)
            p (ndarray): pressure array (for Ozone_normed_pressure)
            norm_new (float): normalisation pressure value
                (for Ozone_normed_pressure)

        Returns:
            ndarray: new ozone profile
        """


class Ozone_pressure(Ozone):
    """Ozone fixed with pressure, no adjustment needed."""
    def get(self, *args, **kwargs):
        return self.initial_ozone


class Ozone_height(Ozone):
    """Ozone fixed with height."""
    def get(self, height_new, *args, **kwargs):
        f = interp1d(self.initial_height, self.initial_ozone,
                     fill_value='extrapolate')
        return f(height_new)


class Ozone_normed_pressure(Ozone):
    """Ozone shifts with the normalisation level (chosen to be the convective
    top)."""
    def get(self, height_new, p, norm_new):
        f = interp1d(p/self.norm_level, self.initial_ozone,
                     fill_value='extrapolate')
        return f(p/norm_new)

