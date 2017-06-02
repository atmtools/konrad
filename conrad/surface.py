# -*- coding: utf-8 -*-
"""Module containing classes describing different surface models.
"""

__all__ = [
    'Surface',
    'SurfaceFixedTemperature',
    'SurfaceAdjustableTemperature',
]


import abc

import numpy as np

from . import constants


class Surface(metaclass=abc.ABCMeta):
    """Abstract base class to define requirements for surface models."""
    def __init__(self, albedo=0.3, temperature=288, pressure=101325):
        """Initialize a surface model.

        Parameters:
            albedo (float): Surface albedo.
            temperature (float): Surface temperature [K].
            pressure (float): Surface pressure [Pa].
        """
        self.albedo = albedo
        self.temperature = temperature
        self.pressure = pressure

    @abc.abstractmethod
    def adjust(self, s, ir):
        """Adjust the surface according to given radiative fluxes.

        Parameters:
            s (float): Shortwave net flux [W / m**2].
            ir (float): Longwave downward flux [W / m**2].
        """
        pass

    @classmethod
    def from_atmosphere(cls, atmosphere, **kwargs):
        """Initialize a Surface object using the lowest atmosphere layer.

        Paramters:
            atmosphere (conrad.atmosphere.Atmosphere): Atmosphere model.
        """

        # Copy temperature of lowst atmosphere layer.
        T_sfc = atmosphere['T'].values[0, 0]

        # Extrapolate surface pressure from last two atmosphere layers.
        p = atmosphere['plev'].values
        p_sfc = p[0] - np.diff(p)[0]

        return cls(temperature=T_sfc,
                   pressure=p_sfc,
                   **kwargs,
                   )


class SurfaceFixedTemperature(Surface):
    """Surface model with fixed temperature."""
    def adjust(self, *args, **kwargs):
        """Do not adjust anything for fixed temperature surfaces.

        This function takes an arbitrary number of positional arguments and
        keyword arguments and does nothing.

        Notes:
            Dummy function to fulfill abstract class requirements.
        """
        pass


class SurfaceAdjustableTemperature(Surface):
    """Surface model with adjustable temperature."""
    def adjust(self, s, ir):
        """Increase the surface temperature by given heatingrate.

        Parameters:
            s (float): Shortwave net flux [W / m**2].
            ir (float): Longwave downward flux [W / m**2].

        Notes:
            The surface is assumed to have no heat capacity.
        """
        self.temperature = ((s + ir) / constants.stefan_boltzmann)**0.25
