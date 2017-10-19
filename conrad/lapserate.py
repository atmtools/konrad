# -*- coding: utf-8 -*-
"""Contains classes for handling atmospheric temperature lapse rates."""
import abc
import numbers

import numpy as np

from conrad import (constants, utils)


class LapseRate(metaclass=abc.ABCMeta):
    """Base class for all lapse rate handlers."""
    @abc.abstractmethod
    def get(self, T, VMR):
        """Return the atmospheric lapse rate.

        Parameters:
              T (ndarray): Atmospheric temperature [K].
              VMR (ndarray): Water vapor content [VMR].

        Returns:
              ndarray: Temperature lapse rate [K/m].
        """


class MoistLapseRate(LapseRate):
    """Moist adiabatic temperature lapse rate."""
    def get(self, T, VMR):
        # Use short forumula symbols for physical constants.
        g = constants.earth_standard_gravity
        Lv = constants.heat_of_vaporization
        R = constants.specific_gas_constant_dry_air
        epsilon = constants.gas_constant_ratio
        Cp = constants.isobaric_mass_heat_capacity

        lapse = (
            g * (1 + Lv * VMR / R / T)
            / (Cp + Lv**2 * VMR * epsilon / R / T**2)
        )

        # TODO (Sally): Replace with correct interpolation.
        lapse_phlev = utils.calculate_halflevel_pressure(lapse)

        return lapse_phlev


class FixedLapseRate(LapseRate):
    """Fixed linear lapse rate through the whole atmosphere."""
    def __init__(self, lapserate=0.0065):
        """Create a handler with fixed linear temperature lapse rate.

        Parameters:
              lapserate (float or ndarray): Critical lapse rate [K/m].
        """
        self.lapserate = lapserate

    def get(self, T, VMR):
        if isinstance(self.lapserate, numbers.Number):
            return self.lapserate * np.ones(T.size + 1)
        elif isinstance(self.lapserate, np.ndarray):
            return self.lapserate
