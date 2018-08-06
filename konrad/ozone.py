# -*- coding: utf-8 -*-
"""This module contains classes handling different treatments of ozone."""

import abc
import logging
from scipy.interpolate import interp1d
from konrad.utils import ozone_profile_rcemip, refined_pgrid

__all__ = [
    'Ozone',
    'OzonePressure',
    'OzoneHeight',
    'OzoneNormedPressure',
]

logger = logging.getLogger(__name__)

class Ozone(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for ozone treatments."""

    def __init__(self, initial_ozone=None):
        """
        Parameters:
            initial_ozone (ndarray): initial ozone vmr profile
        """
        if initial_ozone is None:
            initial_ozone = ozone_profile_rcemip(
                    refined_pgrid(1013e2, 0.01e2, 200))
        self.initial_ozone = initial_ozone

    @abc.abstractmethod
    def get(self, atmos, timestep, zenith, radheat):
        """Updates the ozone profile within the atmosphere class.

        Parameters:
            atmos (konrad.atmosphere): atmosphere model containing ozone
                concentration profile, height, temperature, pressure and half
                pressure levels at the current timestep
            timestep (float): timestep of run [days]
            zenith (float): solar zenith angle,
                angle of the Sun to the vertical [degrees]
            radheat (ndarray): array of net radiative heating rates
        """

class OzonePressure(Ozone):
    """Ozone fixed with pressure, no adjustment needed."""
    def get(self, atmos, **kwargs):
        atmos['O3'].values[0, :] = self.initial_ozone
        return


class OzoneHeight(Ozone):
    """Ozone fixed with height."""
    def __init__(self, initial_height=None, initial_ozone=None, **kwargs):
        """
        Parameters:
            initial_height (ndarray): altitude profile [m]
        """
        super().__init__(initial_ozone=initial_ozone)
        self.initial_height = initial_height

        self.f = interp1d(self.initial_height, self.initial_ozone,
                     fill_value='extrapolate')

    def get(self, atmos, **kwargs):
        z = atmos.get_values('z', keepdims=False)
        atmos['O3'].values[0, :] = self.f(z)
        return


class OzoneNormedPressure(Ozone):
    """Ozone shifts with the normalisation level (chosen to be the convective
    top)."""
    def __init__(self, norm_level=None, initial_ozone=None, **kwargs):
        """
        Parameters:
            norm_level (float): pressure for the normalisation
                normally chosen as the convective top pressure at the start of
                the simulation [Pa]
        """
        super().__init__(initial_ozone=initial_ozone)
        self.norm_level = norm_level
        self.f = None

    def get(self, atmos, radheat, **kwargs):
        p = atmos.get_values('plev')
        norm_new = float(atmos.get_convective_top(radheat))

        if self.f is None:
            self.f = interp1d(p/self.norm_level, self.initial_ozone,
                              fill_value='extrapolate')

        atmos['O3'].values[0, :] = self.f(p/norm_new)
        return

