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
    def adjust(self, heatingrate):
        """Adjust the surface according to given heatingrate.

        Paramters:
            """
        pass

    @classmethod
    def from_atmosphere(cls, atmosphere, **kwargs):
        """Initialize a Surface object using the lowest atmosphere layer.

        Paramters:
            atmosphere (conrad.atmosphere.Atmosphere): Atmosphere model.
        """
        return cls(temperature=atmosphere['T'].values[0, 0],
                   pressure=atmosphere['plev'].values[0],
                   **kwargs,
                   )


class SurfaceFixedTemperature(Surface):
    """Surface model with fixed temperature."""
    def adjust(self, heatingrate):
        """Do not adjust anything for fixed temperature surfaces.

        Notes:
            Dummy function to fulfill abstract class requirements.
        """
        pass


class SurfaceAdjustableTemperature(Surface):
    """Surface model with adjustable temperature."""
    def adjust(self, heatingrate):
        """Increase the surface temperature by given heatingrate.

        Notes:
            The surface is assmued to have no heat capacity.
        """
        self.temperature += heatingrate


class SurfaceCoupled(Surface):
    def __init__(self, albedo=0.3, temperature=288, pressure=101325,
                 atmosphere=None):
        """Initialize a surface model.

        Parameters:
            albedo (float): Surface albedo.
            temperature (float): Surface temperature [K].
            pressure (float): Surface pressure [Pa].
        """
        self.albedo = albedo
        self.temperature = temperature
        self.pressure = pressure
        self.atmosphere = atmosphere

    @classmethod
    def from_atmosphere(cls, atmosphere, **kwargs):
        """Initialize a Surface object using the lowest atmosphere layer.

        Paramters:
            atmosphere (conrad.atmosphere.Atmosphere): Atmosphere model.
        """
        return cls(temperature=atmosphere['T'].values[0, 0],
                   pressure=atmosphere['plev'].values[0],
                   atmosphere=atmosphere,
                   **kwargs,
                   )

    def adjust(self, heatingrates):
        self.temperature = self.atmosphere['T'].values[0, 0]
        self.pressure = self.atmosphere['plev'].values[0]

        print(self.temperature)
