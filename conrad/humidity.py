# -*- coding: utf-8 -*-
"""This module contains classes for handling humidity."""
import abc

import numpy as np
from scipy.stats import skewnorm
from typhon.atmosphere import vmr


class Humidity(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all humidity handlers."""
    def __init__(self, rh_surface=0.8, rh_tropo=0.3, p_tropo=170e2,
                 vmr_strato=None, transition_depth=None):
        """Create a humidity handler.

        Parameters:
            rh_surface (float): Relative humidity at first
                pressure level (surface).
            rh_tropo (float): Relative humidity at second maximum
                in the upper-troposphere.
            p_tropo (float): Pressure level of second humidity maximum [Pa].
            vmr_strato (float): Stratospheric water vapor VMR.
            transition_depth (float): Transition depth from humidity at the
                cold point to the straotpsheric background [m].
        """
        self.rh_surface = rh_surface
        self.rh_tropo = rh_tropo
        self.p_tropo = p_tropo
        self.vmr_strato = vmr_strato
        self.transition_depth = transition_depth

        self.vmr = None  # Attribute may be used for chaching later one.

    def relative_humidity_profile(self, p):
        """Create a realistic relative humidity profile for the tropics.

        Parameters:
            p (ndarray): Pressure [Pa].

        Returns:
            ndarray: Relative humidity.
        """
        # Exponential decay from the surface value throughout the atmosphere.
        rh = self.rh_surface * (p / p[0]) ** 1.15

        # Add skew-normal distribution.
        rh += self.rh_tropo * skewnorm.pdf(
            x=np.log(p),
            # NOTE: Subtract 20e2 hPa to match the passed value of `p_tropo`
            # with the actual peak of the skewnorm distribution.
            loc=np.log(self.p_tropo - 20e2),
            # Shape parameters are fitted to an ERA5 climatology.
            a=4,
            scale=0.75,
        )

        return rh

    def adjust_stratosphere(self, vmr, p, T, z, cold_point_min=1e2):
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
        # If stratospheric background = None, use cold point VMR, otherwise 
        # transition to the stratospheric background value
        if self.vmr_strato == None:
            vmr[cp_index:] = vmr[cp_index]
        else:
            d = self.transition_depth
            if d == None:
                raise ValueError('Specify a transition depth for the ' +
                                 'stratospheric water vapour mixing ratio.')
            vmr_strato = self.vmr_strato
            vmr_cp = vmr[cp_index]
            cpz = z[cp_index] # height of the base of the transition
            
            # index of the top of the transition, this breaks if d is too large
            # and z is nowhere bigger than `cpz + transition_depth`.
            d_i = np.min(np.where(z > cpz+self.transition_depth))
            
            vmr[cp_index:d_i] = (
                -np.cos(np.pi*(z[cp_index:d_i]-cpz)/(self.transition_depth))
                * (vmr_strato - vmr_cp)/2
                + (vmr_strato + vmr_cp)/2
            )
            vmr[d_i:] = vmr_strato

        return vmr

    def vmr_profile(self, p, T, z):
        """Return a water vapor volume mixing ratio profile.

        Parameters:
            p (ndarray): Pressure levels [Pa].
            T (ndarray): Temperature [K].
            z (ndarray): Altitude [m].

        Returns:
            ndarray: Water vapor profile [VMR].
        """
        self.vmr = vmr(self.relative_humidity_profile(p), p, T)
        self.vmr = self.adjust_stratosphere(self.vmr, p, T, z)

        return self.vmr

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


class FixedVMR(Humidity):
    """Keep the water vapor volume mixing ratio constant."""
    def get(self, plev, T, z, **kwargs):
        if self.vmr is None:
            self.vmr = self.vmr_profile(plev, T, z)

        return self.vmr


class FixedRH(Humidity):
    """Preserve the relative humidity profile under temperature changes.

    The relative humidity is kept constant under temperature changes,
    allowing for a moistening in a warming climate.
    """
    def get(self, plev, T, z, **kwargs):
        self.vmr = self.vmr_profile(plev, T, z)

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
    def get(self, plev, T, z, p_tropo=150e2, **kwargs):
        self.p_tropo = p_tropo

        self.vmr = self.vmr_profile(plev, T, z)

        return self.vmr
