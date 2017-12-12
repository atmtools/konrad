# -*- coding: utf-8 -*-
"""Contains classes for handling atmospheric temperature lapse rates."""
import abc
import numbers

import numpy as np
from typhon.physics import (e_eq_water_mk, vmr2specific_humidity)
from scipy.interpolate import interp1d

from conrad import constants


class LapseRate(metaclass=abc.ABCMeta):
    """Base class for all lapse rate handlers."""
    @abc.abstractmethod
    def get(self, atmosphere):
        """Return the atmospheric lapse rate.

        Parameters:
              T (ndarray): Atmospheric temperature [K].
              p (ndarray): Atmospheric pressure [Pa].

        Returns:
              ndarray: Temperature lapse rate [K/m].
        """

class MoistLapseRate2(LapseRate):
    """ Moist adiabatic temperature lapse rate, with
    temperature varying latent heat and no simplification for q_saturated.
    
    Currently behaving strangly near the surface.
    """
    def get(self, atmosphere):
        T_surface = atmosphere.surface.temperature
        p_surface = atmosphere.surface.pressure
        z_surface = atmosphere.surface.height
        T_a = atmosphere['T'][0, :]
        T = np.hstack((T_surface, T_a))
        p = np.hstack((p_surface, atmosphere['plev'][:]))
        z = np.hstack((z_surface, atmosphere['z'][0, :]))
        g = constants.earth_standard_gravity
        Cp = constants.isobaric_mass_heat_capacity
        L = 2500800 - 2360*(T_a-273) + 1.6*(T_a-273)**2 - 0.06*(T_a-273)**3
        #L = constants.heat_of_vaporization
        gamma_d = g / Cp
        q_saturated = vmr2specific_humidity(e_eq_water_mk(T) / p)
        dqdz = np.diff(q_saturated) / np.diff(z)
        gamma_m = gamma_d + L/Cp*dqdz
        gamma_m[np.min(np.where(gamma_m > gamma_d)):] = gamma_d
        return gamma_m

class MoistLapseRate(LapseRate):
    """Moist adiabatic temperature lapse rate."""
    def get(self, atmosphere):
        T = atmosphere['T'][0, :]
        p = atmosphere['plev'][:]
        phlev = atmosphere['phlev'][:]

        # Use short formula symbols for physical constants.
        g = constants.earth_standard_gravity
        L = constants.heat_of_vaporization
        Rd = constants.specific_gas_constant_dry_air
        Rv = constants.specific_gas_constant_water_vapor
        Cp = constants.isobaric_mass_heat_capacity

        gamma_d = g / Cp  # dry lapse rate
        q_saturated = vmr2specific_humidity(e_eq_water_mk(T) / p)

        gamma_m = (gamma_d * ((1 + (L * q_saturated) / (Rd * T)) /
                              (1 + (L**2 * q_saturated) / (Cp * Rv * T**2))
                              )
        )
        lapse = interp1d(p, gamma_m, fill_value='extrapolate')(phlev[:-1])
        return lapse


class FixedLapseRate(LapseRate):
    """Fixed linear lapse rate through the whole atmosphere."""
    def __init__(self, lapserate=0.0065):
        """Create a handler with fixed linear temperature lapse rate.

        Parameters:
              lapserate (float or ndarray): Critical lapse rate [K/m].
        """
        self.lapserate = lapserate

    def get(self, atmosphere):
        if isinstance(self.lapserate, numbers.Number):
            T = atmosphere['T'][0, :]
            return self.lapserate * np.ones(T.size)
        elif isinstance(self.lapserate, np.ndarray):
            return self.lapserate
