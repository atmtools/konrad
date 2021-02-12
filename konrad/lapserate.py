# -*- coding: utf-8 -*-
"""Contains classes for handling atmospheric temperature lapse rates.
These are used by the convection sub-model to set the temperature profile in
the troposphere.

**Example**

Calculate the moist adiabatic lapse rate, for a specified atmospheric
temperature profile:

    >>> import konrad
    >>> lapse = konrad.lapserate.MoistLapseRate()
    >>> critical_lapserate = lapse(atmosphere=...)

Apply convection to an unstable atmosphere, updating the atmospheric temperature
profile and the surface temperature to follow the :code:`critical_lapserate`:

    >>> convection = konrad.convection.HardAdjustment()
    >>> convection.stabilize(
    >>>     atmosphere=..., surface=..., lapse=critical_lapserate)

"""
import abc

import numpy as np
from typhon.physics import vmr2mixing_ratio
from scipy.interpolate import interp1d

from konrad import constants
from konrad.component import Component
from konrad.physics import saturation_pressure


class LapseRate(Component, metaclass=abc.ABCMeta):
    """Base class for all lapse rate handlers."""
    @abc.abstractmethod
    def __call__(self, p, T):
        """Return the atmospheric lapse rate.

        Parameters:
              p (ndarray): Atmospheric pressure [Pa].
              T (ndarray): Atmospheric temperature [K].

        Returns:
              ndarray: Temperature lapse rate [K/m].
        """


class MoistLapseRate(LapseRate):
    """Moist adiabatic temperature lapse rate."""
    def __init__(self, fixed=False):
        self._lapse_cache = None

        if fixed:
            raise ValueError(
                "The `fixed` keyword is no longer supported.\n"
                "Use `konrad.lapserate.MoistLapseRate.build_cache()` instead."
            )

    def __call__(self, p, T):
        # Use cached lapse rate if present, otherwise calculate it.
        if self._lapse_cache is not None:
            return self._lapse_cache(np.log(p))
        else:
            return self.calc_lapse_rate(p, T)

    def build_cache(self, atmosphere):
        """Build a lapse-rate cache from a given atmospheric state."""
        p = atmosphere["plev"]
        T = atmosphere["T"][-1]

        self._lapse_cache = interp1d(
            np.log(p),
            self.calc_lapse_rate(p, T),
            kind="linear",
            fill_value="extrapolate",
        )

    def calc_lapse_rate(self, p, T):
        # Use short formula symbols for physical constants.
        g = constants.earth_standard_gravity
        L = constants.heat_of_vaporization
        Rd = constants.specific_gas_constant_dry_air
        Rv = constants.specific_gas_constant_water_vapor
        Cp = constants.isobaric_mass_heat_capacity_dry_air

        gamma_d = g / Cp  # dry lapse rate

        w_saturated = vmr2mixing_ratio(saturation_pressure(T) / p)

        gamma_m = (gamma_d * ((1 + (L * w_saturated) / (Rd * T)) /
                              (1 + (L**2 * w_saturated) / (Cp * Rv * T**2))
                              )
                   )

        return gamma_m


class FixedLapseRate(LapseRate):
    """Fixed constant lapse rate through the whole atmosphere. Linear decrease
    in temperature with height."""
    def __init__(self, lapserate=0.0065):
        """
        Parameters:
              lapserate (float or ndarray): Critical lapse rate [K/m].
        """
        self.lapserate = lapserate

    def __call__(self, p, T):
        return self.lapserate
