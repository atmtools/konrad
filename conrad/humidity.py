# -*- coding: utf-8 -*-
"""This module contains classes for handling humidity."""
import abc

import numpy as np
from typhon.atmosphere import vmr


class Humidity(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all humidity handlers."""
    def __init__(self, rh_surface=0.8, rh_tropo=0.3, p_tropo=150e2, c=1.7):
        """Create a humidity handler.

        Parameters:
            rh_surface (float): Relative humidity at first
                pressure level (surface).
            rh_tropo (float): Relative humidity at second maximum
                in the upper-troposphere.
            p_tropo (float): Pressure level of second humidity maximum [Pa].
            c (float): Factor to control the width of the
                upper-tropospheric peak.
        """
        self.rh_surface = rh_surface
        self.rh_tropo = rh_tropo
        self.p_tropo = p_tropo
        self.c = c
        self.vmr = None

    def relative_humidity_profile(self, p):
        """Create a realistic relative humidity profile for the tropics.

        Parameters:
            p (ndarray): Pressure [Pa].

        Returns:
            ndarray: Relative humidity.
        """
        # Exponential decay from the surface value throughout the atmosphere.
        rh = (self.rh_surface / (np.exp(1) - 1)
              * (np.exp((p / p[0]) ** 1.1) - 1)
        )

        # Add  Gaussian centered at a given pressure in the upper troposhere.
        rh += self.rh_tropo * np.exp(-self.c * (np.log(p / self.p_tropo) ** 2))

        return rh

    def adjust_stratosphere(self, vmr, p, T, cold_point_min=1e2):
        """Adjust water vapor VMR values in the stratosphere.

        The stratosphere is determined using the cold point (pressure level
        with lowest temperatures above a given pressure threshold).

        Parameters:
            vmr (ndarray): Water vapor mixing ratio [VMR].
            p (ndarray): Pressure levels [Pa].
            T (ndarray): Temperature [K].
            cold_point_min (float): Lower threshold for cold point pressure.

        Returns:
              ndarray: Adjusted water vapor profile [VMR].
        """
        # Determine the level index of the cold point as proxy for the
        # transition between troposphere and stratosphere. The pressure has
        # to be above a given threshold to prevent finding a cold point at
        # the top of the atmosphere.
        cp_index = int(np.argmin(T[p > cold_point_min]))

        # Keep water vapor VMR values above the cold point tropopause constant.
        vmr[cp_index:] = vmr[cp_index]

        return vmr

    def vmr_profile(self, p, T):
        """Return a water vapor volume mixing ratio profile.

        Parameters:
            p (ndarray): Pressure levels [Pa].
            T (ndarray): Temperature [K].

        Returns:
            ndarray: Water vapor profile [VMR].
        """
        self.vmr = vmr(self.relative_humidity_profile(p), p, T)
        self.vmr = self.adjust_stratosphere(self.vmr, p, T)

        return self.vmr

    @abc.abstractmethod
    def determine(self, plev, T, **kwargs):
        """Determine the humidity profile based on atmospheric state.

        Parameters:
            p (ndarray): Pressure levels [Pa].
            T (ndarray): Temperature [K].

        Returns:
            ndarray: Water vapor profile [VMR].
        """


class FixedVMR(Humidity):
    """Keep the water vapor volume mixing ratio constant."""
    def determine(self, plev, T, **kwargs):
        if self.vmr is None:
            self.vmr = self.vmr_profile(plev, T)

        return self.vmr


class FixedRH(Humidity):
    """Preserve the relative humidity profile under temperature changes.

    The relative humidity is kept constant under temperature changes,
    allowing for a moistening in a warming climate.
    """
    def determine(self, plev, T, **kwargs):
        self.vmr = self.vmr_profile(plev, T)

        return self.vmr


class CoupledRH(Humidity):
    """Couple the relative humidity profile to the top of convection.

    The relative humidity is kept constant under temperature changes,
    allowing for a moistening in a warming climate. In addition,
    the vertical structure of the humidity profile is coupled to the
    structure of atmospheric convection (Zelinke et al, 2010).

    References:
        Zelinka and Hartmann, 2010, Why is longwave cloud feedback positive?
    """
    def determine(self, plev, T, p_tropo=150e2, **kwargs):
        self.p_tropo = p_tropo

        self.vmr = self.vmr_profile(plev, T)

        return self.vmr
