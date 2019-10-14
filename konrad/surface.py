# -*- coding: utf-8 -*-
"""This module contains classes describing different surfaces. A surface is
required by both the radiation and the convective models used inside the RCE
simulations.

**Example**

Create a surface model, *e.g.* :py:class:`SurfaceHeatCapacity`,
and use it in an RCE simulation.
    >>> import konrad
    >>> surface_temperature_start = ...
    >>> surface = konrad.surface.SurfaceHeatCapacity(
    >>>     temperature=surface_temperature_start)
    >>> rce = konrad.RCE(atmosphere=..., surface=surface)
    >>> rce.run()
    >>> surface_temperature_end = surface['temperature'][-1]
"""
import abc
import logging

import netCDF4
import numpy as np

from . import constants
from konrad.component import Component


__all__ = [
    'Surface',
    'SurfaceFixedTemperature',
    'SurfaceHeatCapacity',
    'SurfaceHeatSink',
]


logger = logging.getLogger(__name__)


class Surface(Component, metaclass=abc.ABCMeta):
    """Abstract base class to define requirements for surface models."""
    def __init__(self, albedo=0.2, temperature=288., longwave_emissivity=1,
                 height=0.):
        """Initialize a surface model.

        Parameters:
            albedo (float): Surface albedo. The default value of 0.2 is a
                decent choice for clear-sky simulation in the tropics.
            temperature (int / float): Surface temperature [K].
            longwave_emissivity (float): Longwave emissivity.
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
        """Initialize a Surface object using the lowest atmosphere layer.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
        """
        # Extrapolate surface height from geopotential height of lowest two
        # atmospheric layers.
        atmosphere.calculate_height()
        z = atmosphere['z'][0, :]
        z_sfc = z[0] + 0.5 * (z[0] - z[1])

        # Calculate the surface temperature following a linear lapse rate.
        # This prevents "jumps" after the first iteration, when the
        # convective adjustment is applied.
        # TODO: Perform linear or quadratic interpolation of T profile.
        lapse = 0.0065
        t_sfc = atmosphere['T'][0, 0] + lapse * (z[0] - z_sfc)

        return cls(temperature=t_sfc,
                   height=z_sfc,
                   **kwargs,
                   )

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

            t = dataset['temperature'][timestep].data
            z = float(dataset['height'][:])

        # TODO: Should other variables (e.g. albedo) also be read?
        return cls(temperature=t, height=z, **kwargs)


#TODO: Rename `FixedTemperature`?
class SurfaceFixedTemperature(Surface):
    """Surface model with fixed temperature."""
    def adjust(self, *args, **kwargs):
        """Do not adjust anything for fixed temperature surfaces.

        This function takes an arbitrary number of positional arguments and
        keyword arguments and does nothing.

        Notes:
            Dummy function to fulfill abstract class requirements.
        """
        return


#TODO: Rename `SlabOcean`?
class SurfaceHeatCapacity(Surface):
    """Surface model with adjustable temperature."""
    def __init__(self, *args, depth=50., **kwargs): 
        """
        Parameters:
            depth (float): Ocean depth [m].
            albedo (float): Surface albedo, default 0.2
            temperature (float): Surface temperature [K], default 288
            height (float): Surface height [m], default 0
        """
        super().__init__(*args, **kwargs)
        self.rho = constants.density_sea_water
        self.c_p = constants.specific_heat_capacity_sea_water
        self.depth = depth

        self.heat_capacity = self.rho * self.c_p * depth

    def adjust(self, sw_down, sw_up, lw_down, lw_up, timestep):
        """Increase the surface temperature by given heatingrate.

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

        self['temperature'] += (timestep * net_flux / self.heat_capacity)

        logger.debug("Surface temperature: {self['temperature'][0]:.4f} K")


class SurfaceHeatSink(SurfaceHeatCapacity):
    """Surface model with adjustable temperature."""
    def __init__(self, *args, heat_flux=0, **kwargs):
        """
        Parameters:
            heat_flux(float): Flux of energy out of the surface [W m^-2]
            depth (float): Ocean depth [m], default 50
            albedo (float): Surface albedo, default 0.2
            temperature (float): Surface temperature [K], default 288
            height (float): Surface height [m], default 0
        """
        super().__init__(*args, **kwargs)
        self.heat_flux = heat_flux

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
        sink = self.heat_flux

        logger.debug(f'Net flux: {net_flux:.2f} W /m^2')

        self['temperature'] += (timestep * (net_flux - sink) /
                                self.heat_capacity)

        logger.debug("Surface temperature: {self['temperature'][0]:.4f} K")
