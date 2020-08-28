# -*- coding: utf-8 -*-
"""This module contains classes describing different surfaces.

A surface is required by both the radiation and the convective models used
inside the RCE simulations, and if you don't set it up default will be a `SlabOcean`.

**Example**

Create a surface model, *e.g.* :py:class:`SlabOcean`,
and use it in an RCE simulation:

    >>> import konrad
    >>> surface_temperature_start = ...
    >>> surface = konrad.surface.SlabOcean(
    >>>     temperature=surface_temperature_start)
    >>> rce = konrad.RCE(atmosphere=..., surface=surface)
    >>> rce.run()
    >>> surface_temperature_end = surface['temperature'][-1]

"""
import abc
import logging

import netCDF4
import numpy as np
from scipy.interpolate import interp1d

from . import constants
from konrad.component import Component


__all__ = [
    'Surface',
    'FixedTemperature',
    'SlabOcean',
]


logger = logging.getLogger(__name__)


class Surface(Component, metaclass=abc.ABCMeta):
    """Abstract base class to define requirements for surface models."""
    def __init__(self, albedo=0.2, temperature=288., longwave_emissivity=1,
                 height=0.):
        """Initialize a surface model.

        Parameters:
            albedo (float): Surface albedo [1]. The default value of 0.2 is a
                decent choice for clear-sky simulation in the tropics.
            temperature (int / float): Surface temperature [K].
            longwave_emissivity (float): Longwave emissivity [1].
            height (int / float): Surface height [m].
        """
        self.albedo = albedo
        self.longwave_emissivity = longwave_emissivity
        self.height = height
        self['temperature'] = (('time',), np.array([temperature], dtype=float))

        # The surface pressure is initialized before the first iteration
        # within the RCE framework to ensure a pressure that is consistent
        # with the atmosphere used.
        self.pressure = None

        self.coords = {
            'time': np.array([]),
        }

    @abc.abstractmethod
    def adjust(self, sw_down, sw_up, lw_down, lw_up, timestep):
        """Adjust the surface according to given radiative fluxes.

        Parameters:
            sw_down (float): Shortwave downward flux [W / m**2].
            sw_up (float): Shortwave upward flux [W / m**2].
            lw_down (float): Longwave downward flux [W / m**2].
            lw_up (float): Longwave upward flux [W / m**2].
            timestep (float): Timestep in days.
        """
        pass

    @classmethod
    def from_atmosphere(cls, atmosphere, **kwargs):
        """Initialize the surface by extrapolating the atmospheric temperature.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
        """
        f = interp1d(
            x=atmosphere['plev'],
            y=atmosphere['T'][-1],
            kind="cubic",
            fill_value="extrapolate",
        )

        # The surface is placed at the lowest half-level pressure (`phlev`).
        return cls(temperature=f(atmosphere['phlev'][0]), **kwargs)

    @classmethod
    def from_netcdf(cls, ncfile, timestep=-1, **kwargs):
        """Create a surface model from a netCDF file.

        Parameters:
            ncfile (str): Path to netCDF file.
            timestep (int): Timestep to read (default is last timestep).
        """
        with netCDF4.Dataset(ncfile) as root:
            if 'surface' in root.groups:
                dataset = root['surface']
            else:
                dataset = root

            t = dataset['temperature'][timestep].data.astype("float64")
            z = float(dataset['height'][:])
            alb = float(dataset['albedo'][:])
            le = float(dataset['longwave_emissivity'][:])

        return cls(temperature=t, height=z, albedo=alb, longwave_emissivity=le, **kwargs)


class SlabOcean(Surface):
    """Surface model with adjustable temperature."""
    def __init__(self, *args, depth=1.0, heat_sink=66.0, **kwargs):
        """Initialize a slab ocean.

        Parameters:
            heat_sink (float): Flux of energy out of the surface [W m^-2].
                The default value represents a surface enthalpy transport to
                the extra-tropics.
            depth (float): Ocean depth [m].
            albedo (float): Surface albedo [1].
            temperature (float): Initial surface temperature [K].
            longwave_emissivity (float): Longwave emissivity [1].
            height (float): Surface height [m].
        """
        super().__init__(*args, **kwargs)

        self.rho = constants.density_sea_water
        self.c_p = constants.specific_heat_capacity_sea_water
        self.depth = depth

        self.heat_capacity = self.rho * self.c_p * depth
        self.heat_sink = heat_sink

    def adjust(self, sw_down, sw_up, lw_down, lw_up, timestep):
        """Increase the surface temperature using given radiative fluxes. Take
        into account a heat sink at the surface, as if heat is transported out
        of the tropics we are modelling.

        Parameters:
            sw_down (float): Shortwave downward flux [W / m**2].
            sw_up (float): Shortwave upward flux [W / m**2].
            lw_down (float): Longwave downward flux [W / m**2].
            lw_up (float): Longwave upward flux [W / m**2].
            timestep (float): Timestep in days.
        """
        timestep *= 24 * 60 * 60  # Convert timestep to seconds.

        net_flux = (sw_down - sw_up) + (lw_down - lw_up)

        logger.debug(f'Net flux: {net_flux:.2f} W /m^2')

        self['temperature'] += (timestep * (net_flux - self.heat_sink) /
                                self.heat_capacity)

        logger.debug("Surface temperature: {self['temperature'][0]:.4f} K")

    @classmethod  
    def from_netcdf(cls, ncfile, timestep=-1, **kwargs):
        """Create a surface model from a netCDF file.

        Parameters:
            ncfile (str): Path to netCDF file.
            timestep (int): Timestep to read (default is last timestep).
        """
        with netCDF4.Dataset(ncfile) as root:
            if 'surface' in root.groups:
                dataset = root['surface']
            else:
                dataset = root

            t = dataset['temperature'][timestep].data.astype("float64")
            z = float(dataset['height'][:])
            alb = float(dataset['albedo'][:])
            le = float(dataset['longwave_emissivity'][:])
            hs = float(dataset['heat_sink'][:])
            d = float(dataset['depth'][:])
            
        return cls(temperature=t, height=z, albedo=alb, longwave_emissivity=le, heat_sink=hs, depth=d, **kwargs)


class FixedTemperature(Surface):
    """Surface model with fixed temperature."""
    def __init__(self, *args, **kwargs):
        """Initialize a surface model with constant surface temperature.

        Parameters:
            albedo (float): Surface albedo. The default value of 0.2 is a
                decent choice for clear-sky simulation in the tropics.
            temperature (int / float): Surface temperature [K].
            longwave_emissivity (float): Longwave emissivity.
            height (int / float): Surface height [m].
        """
        super().__init__(*args, **kwargs)
        self.heat_capacity = np.inf

    def adjust(self, *args, **kwargs):
        """Do not adjust anything for fixed temperature surfaces.

        This function takes an arbitrary number of positional arguments and
        keyword arguments and does nothing.

        Notes:
            Dummy function to fulfill abstract class requirements.
        """
        return
