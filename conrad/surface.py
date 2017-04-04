# -*- coding: utf-8 -*-
"""Module containing classes describing different surface models.
"""

__all__ = [
    'Surface',
    'SurfaceFixedTemperature',
    'SurfaceAdjustableTemperature',
]


import abc


class Surface(metaclass=abc.ABCMeta):
    """Abstract base class to define common requirements for surface models."""
    def __init__(self, albedo=0.3, temperature=288, pressure=101325):
        """Create an surface model object.

        Parameters:
            albedo (float): Surface albedo.
            temperature (float): Surface temperature [K].
            pressure (float): Surface pressure [Pa].
        """
        self.albedo = albedo
        self.temperature = temperature
        self.pressure = pressure

    @abc.abstractmethod
    def adjust(self, heatingrate):
        """Adjust the surface according to given heatingrate."""
        pass

    @classmethod
    def from_atmosphere(cls, atmosphere):
        """Initialize a Surface object using the lowest atmosphere layer."""
        return cls(temperature=atmosphere['T'].values[0, 0],
                   pressure=atmosphere['plev'].values[0])


class SurfaceFixedTemperature(Surface):
    """Describes a surface with fixed temperature."""
    def adjust(self, heatingrate):
        """Dummy function to fulfill abstract class requirements.

        Do not adjust anything for fixed temperature surfaces.
        """
        return


class SurfaceAdjustableTemperature(Surface):
    """Describes a surface with adjustable temperature."""
    def adjust(self, heatingrate):
        """Increase the surface temperature by given heatingrate.

        The surface is assmued to have no heat capacity.
        """
        self.temperature += heatingrate
