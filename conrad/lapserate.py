# -*- coding: utf-8 -*-
"""Contains classes for handling atmospheric temperature lapse rates."""
import abc
import numbers

import numpy as np
from typhon.physics import e_eq_water_mk

from conrad import constants


class LapseRate(metaclass=abc.ABCMeta):
    """Base class for all lapse rate handlers."""
    @abc.abstractmethod
    def get(self, T, p):
        """Return the atmospheric lapse rate.

        Parameters:
              T (ndarray): Atmospheric temperature [K].
              p (ndarray): Atmospheric pressure [Pa].

        Returns:
              ndarray: Temperature lapse rate [K/m].
        """


class MoistLapseRate(LapseRate):
    """Moist adiabatic temperature lapse rate."""
    def get(self, T, p):
        # Use short formula symbols for physical constants.
        g = constants.earth_standard_gravity
        L = constants.heat_of_vaporization
        Rd = constants.specific_gas_constant_dry_air
        Rv = constants.specific_gas_constant_water_vapor
        epsilon = constants.gas_constant_ratio
        Cp = constants.isobaric_mass_heat_capacity

        gamma_d = g / Cp  # dry lapse rate
        q_saturated = epsilon * e_eq_water_mk(T) / p


        gamma_m = (gamma_d * ((1 + (L * q_saturated) / (Rd * T)) /
                              (1 + (L**2 * q_saturated) / (Cp * Rv * T**2))
                              )
        )

        return gamma_m


class FixedLapseRate(LapseRate):
    """Fixed linear lapse rate through the whole atmosphere."""
    def __init__(self, lapserate=0.0065):
        """Create a handler with fixed linear temperature lapse rate.

        Parameters:
              lapserate (float or ndarray): Critical lapse rate [K/m].
        """
        self.lapserate = lapserate

    def get(self, T, p):
        if isinstance(self.lapserate, numbers.Number):
            return self.lapserate * np.ones(T.size)
        elif isinstance(self.lapserate, np.ndarray):
            return self.lapserate
