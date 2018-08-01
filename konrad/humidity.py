# -*- coding: utf-8 -*-
"""This module contains classes for handling humidity."""
import abc
import logging

import numpy as np
from typhon.atmosphere import vmr as rh2vmr
from scipy.stats import (norm, skewnorm)


logger = logging.getLogger(__name__)


class Humidity(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all humidity handlers."""
    def __init__(self, rh_surface=0.85, rh_tropo=0.4, p_tropo=170e2, offset=0,
                 vmr_strato=None, vmr_profile=None, rh_profile=None):
        """Create a humidity handler.

        Parameters:
            rh_surface (float): Relative humidity at first
                pressure level (surface).
            rh_tropo (float): Relative humidity at second maximum
                in the upper-troposphere.
            p_tropo (float): Pressure level of second humidity maximum [Pa].
            vmr_strato (float): Stratospheric water vapor VMR.
            vmr_profile (ndarray): Water vapour volume mixing ratio profile.
            rh_profile (ndarray): Relative humidity profile.
                Is ignored if the `vmr_profile` is given.
        """
        self.rh_surface = rh_surface
        self.rh_tropo = rh_tropo
        self.p_tropo = p_tropo
        self.offset = offset
        self.vmr_strato = vmr_strato

        self.vmr_profile = vmr_profile
        self.rh_profile = rh_profile

    @abc.abstractmethod
    def get(self, plev, T, z, **kwargs):
        """Determine the humidity profile based on atmospheric state.

        Parameters:
            plev (ndarray): Pressure levels [Pa].
            T (ndarray): Temperature [K].
            z (ndarray): Height [m].

        Returns:
            ndarray: Water vapor profile [VMR].
        """

    def get_relative_humidity_profile(self, p):
        """Create a realistic relative humidity profile for the tropics.

        Parameters:
            p (ndarray): Pressure [Pa].

        Returns:
            ndarray: Relative humidity.
        """
        # If set, return prescribed relative humidity profile.
        if self.rh_profile is not None:
            return self.rh_profile

        # Exponential decay from the surface value throughout the atmosphere.
        rh = self.rh_surface * (p / p[0]) ** 1.25

        # Add skew-normal distribution.
        uth = self.rh_tropo * norm.pdf(
            x=np.log(p),
            loc=np.log(self.p_tropo + self.offset),
            scale=0.4,
        )

        return np.maximum(rh, uth)

    def get_vmr_profile(self, p, T, z):
        """Return a water vapor volume mixing ratio profile.

        Parameters:
            p (ndarray): Pressure levels [Pa].
            T (ndarray): Temperature [K].
            z (ndarray): Altitude [m].

        Returns:
            ndarray: Water vapor profile [VMR].
        """
        if self.vmr_profile is None:
            vmr = rh2vmr(self.get_relative_humidity_profile(p), p, T)
        else:
            vmr = self.vmr_profile

        vmr = self.adjust_stratospheric_vmr(vmr, p, T, z)

        return vmr

    def adjust_stratospheric_vmr(self, vmr, p, T, z, cold_point_min=1e2):
        """Adjust water vapor VMR values in the stratosphere.

        The stratosphere is determined using the cold point (pressure level
        with lowest temperatures above a given pressure threshold).

        Parameters:
            vmr (ndarray): Water vapor mixing ratio [VMR].
            p (ndarray): Pressure levels [Pa].
            T (ndarray): Temperature [K].
            z (ndarray): Height [m].
            cold_point_min (float): Lower threshold for cold point pressure.

        Returns:
              ndarray: Adjusted water vapor profile [VMR].
        """
        # Determine the level index of the cold point as proxy for the
        # transition between troposphere and stratosphere. The pressure has
        # to be above a given threshold to prevent finding a cold point at
        # the top of the atmosphere.
        cp_index = int(np.argmin(T[p > cold_point_min]))

        # Keep stratospheric water vapor VMR values constant.
        # If stratospheric background = None, use cold point VMR, otherwise
        # transition to the stratospheric background value
        if self.vmr_strato is None:
            vmr[cp_index:] = vmr[cp_index]
        else:
            # If the VMR falls below the stratospheric background...
            if np.any(vmr[p > cold_point_min] < self.vmr_strato):
                # ... set all values equal to the background from there.
                vmr[np.argmax(vmr < self.vmr_strato):] = self.vmr_strato
            else:
                # Otherwise find the smallest VMR and use the background value
                # from there on, this at least minimizes the discontinuity at
                # the transition point.
                vmr[np.argmin(vmr[p > cold_point_min]):] = self.vmr_strato

        return vmr


class FixedVMR(Humidity):
    """Keep the water vapor volume mixing ratio constant."""
    def get(self, plev, T, z, **kwargs):
        if self.vmr_profile is None:
            self.vmr_profile = self.get_vmr_profile(plev, T, z)

        return self.get_vmr_profile(plev, T, z)


class FixedRH(Humidity):
    """Preserve the relative humidity profile under temperature changes.

    The relative humidity is kept constant under temperature changes,
    allowing for a moistening in a warming climate.
    """
    def get(self, plev, T, z, **kwargs):
        return self.get_vmr_profile(plev, T, z)


class Manabe67(Humidity):
    """Relative humidity model following Manabe and Wetherald (1967)."""
    def __init__(self, rh_surface=0.77, vmr_strato=4.8e-6):
        """Initialize a humidity model.

        Parameters:
            rh_surface (float): Relative humidity at the surface.
            vmr_strato (float): Minimum water vapor volume mixing ratio.
                Values below this threshold are clipped. The default value
                of `4.8e-6` resembles the *mass* mixing ratio of 3 ppm given
                in the literature.
        """
        super().__init__(rh_surface=rh_surface, vmr_strato=vmr_strato)

    def get_relative_humidity_profile(self, p):
        return self.rh_surface * (p / p[0] - 0.02) / (1 - 0.02)

    def get(self, plev, T, z, **kwargs):
        return self.get_vmr_profile(plev, T, z)


class Cess76(FixedRH):
    """Relative humidity model following Cess (1976).

    The relative humidity profile depends on the surface temperature.
    This results in moister atmospheres at warmer temperatures.
    """
    def __init__(self, rh_surface=0.8, T_surface=288):
        """Initialize a humidity model.

        Parameters:
            rh_surface (float): Relative humidity at the surface.
            T_surface (float): Surface temperature [K].
        """
        super().__init__(rh_surface=rh_surface, rh_tropo=0.)

        self.T_surface = T_surface

    @property
    def omega(self):
        """Temperature dependent scaling factor for the RH profile."""
        return 1.0 - 0.03 * (self.T_surface - 288)

    def get_relative_humidity_profile(self, p):
        return self.rh_surface * ((p / p[0] - 0.02) / (1 - 0.02))**self.omega

    def get(self, plev, T, z, T_surface=288, **kwargs):
        """Determine the humidity profile based on atmospheric state.

        Parameters:
            plev (ndarray): Pressure levels [Pa].
            T (ndarray): Temperature [K].
            z (ndarray): Height [m].
            T_surface (float): Surface temperature [K].

        Returns:
            ndarray: Water vapor profile [VMR].
        """
        self.T_surface = T_surface

        return self.get_vmr_profile(plev, T, z)


class CoupledRH(Humidity):
    """Couple the relative humidity profile to the top of convection.

    The relative humidity is kept constant under temperature changes,
    allowing for a moistening in a warming climate. In addition,
    the vertical structure of the humidity profile is coupled to the
    structure of atmospheric convection (Zelinka et al, 2010).

    References:
        Zelinka and Hartmann, 2010, Why is longwave cloud feedback positive?
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This ensures a proper coupling of the relative humidity profile.
        for attr in ('rh_profile', 'vmr_profile'):
            if getattr(self, attr) is not None:
                logger.warning(
                    'Set attribute "{}" to `None` for coupled '
                    'humidity calculations.'.format(attr)
                )
                setattr(self, attr, None)

    def get(self, plev, T, z, p_tropo=None, **kwargs):
        """Determine the humidity profile based on atmospheric state.

        Parameters:
            plev (ndarray): Pressure levels [Pa].
            T (ndarray): Temperature [K].
            z (ndarray): Height [m].
            p_tropo (float): Pressure level of second humidity maximum [Pa].

        Returns:
            ndarray: Water vapor profile [VMR].
        """
        if p_tropo is not None:
            self.p_tropo = p_tropo

        return self.get_vmr_profile(plev, T, z)
