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
from typhon.physics import vmr2mixing_ratio, density
from scipy.integrate import ode
from scipy.interpolate import interp1d

from konrad import constants
from konrad.component import Component
from konrad.physics import saturation_pressure


def _to_p_coordinates(gamma, p, T):
    """Convert dT/dz(p, T) to dT/dP(p, T)."""
    g = constants.earth_standard_gravity
    rho = density(p, T)

    return gamma / (g * rho)


class LapseRate(Component, metaclass=abc.ABCMeta):
    """Base class for all lapse rate handlers."""

    @abc.abstractmethod
    def __call__(self, p, T):
        """Return the atmospheric lapse rate.

        Parameters:
              p (ndarray): Atmospheric pressure [Pa].
              T (ndarray): Atmospheric temperature [K].

        Returns:
              ndarray: Temperature lapse rate [K/hPa].
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

        gamma_m = gamma_d * (
            (1 + (L * w_saturated) / (Rd * T))
            / (1 + (L ** 2 * w_saturated) / (Cp * Rv * T ** 2))
        )

        return _to_p_coordinates(gamma_m, p, T)


class FixedLapseRate(LapseRate):
    """Fixed constant lapse rate through the whole troposphere. Linear decrease
    in temperature with height."""

    def __init__(self, lapserate=0.0065):
        """
        Parameters:
              lapserate (float or ndarray): Critical lapse rate [K/m].
        """
        self.lapserate = lapserate

    def __call__(self, p, T):
        return _to_p_coordinates(self.lapserate, p, T)


class DryLapseRate(FixedLapseRate):
    """Fixed dry-adiabatic lapse rate through the whole atmosphere."""

    def __init__(self):
        g = constants.earth_standard_gravity
        c_p = constants.isobaric_mass_heat_capacity_dry_air
        gamma_d = g / c_p

        super().__init__(lapserate=gamma_d)


def get_moist_adiabat(p, p_s=None, T_s=300.0, T_min=155.0):
    """Create a moist-adiabat from a given surface T up to the cold point.

    Warning:
        For very high surface temperatures (>320 K) the pressure grid needs a
        high resolution, otherwise the integration becomes unstable due to the
        large lapse-rate at low pressures.

    Parameters:
        p (ndarray): Pressure levels [Pa].
        p_s (float): Surface pressure [Pa].
        T_s (float): Surface temperautre [K].
        T_min (float): Cold-point temperature (constant temperature above).

    Returns:
        ndarray: Moist-adiabativ temperature profile.

    """
    dTdp = MoistLapseRate().calc_lapse_rate

    T = np.zeros_like(p)
    dp = np.gradient(p)

    r = ode(dTdp).set_integrator("lsoda", atol=1e-4)
    r.set_initial_value(T_s, p[0] - 0.5 * dp[0] if p_s is None else p_s)

    i = 0
    while r.successful() and (r.t > p.min() and r.y[0] > T_min):
        r.integrate(p[i])
        T[i] = r.y[0]
        i += 1

    return T.clip(min=T_min)
