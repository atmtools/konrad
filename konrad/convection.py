"""This module contains a choice of convective adjustments, which can be used
in the RCE simulations.

**Example**

Create an instance of a convective adjustment class, *e.g.* the relaxed
adjustment class, and use it in an RCE simulation:

    >>> import konrad
    >>> relaxed_convection=konrad.convection.RelaxedAdjustment()
    >>> rce = konrad.RCE(atmosphere=..., convection=relaxed_convection)
    >>> rce.run()

Currently there are two convective classes that can be used,
:py:class:`HardAdjustment` and :py:class:`RelaxedAdjustment`, and one class
which can be used and does nothing, :py:class:`NonConvective`.
"""
import abc

import numpy as np
import typhon
from scipy.interpolate import interp1d
from scipy.integrate import ode

from konrad import constants
from konrad.component import Component
from konrad.entrainment import NoEntrainment
from konrad.surface import FixedTemperature


__all__ = [
    "energy_difference",
    "latent_heat_difference",
    "interp_variable",
    "Convection",
    "NonConvective",
    "HardAdjustment",
    "FixedDynamicalHeating",
]


def energy_difference(T_2, T_1, sst_2, sst_1, phlev, eff_Cp_s):
    """
    Calculate the energy difference between two atmospheric profiles (2 - 1).

    Parameters:
        T_2: atmospheric temperature profile (2)
        T_1: atmospheric temperature profile (1)
        sst_2: surface temperature (2)
        sst_1: surface temperature (1)
        phlev: pressure half-levels [Pa]
            must be the same for both atmospheric profiles
        eff_Cp_s: effective heat capacity of surface
    """
    Cp = constants.c_pd
    g = constants.g

    dT = T_2 - T_1  # convective temperature change of atmosphere
    dT_s = sst_2 - sst_1  # of surface

    term_diff = -np.sum(Cp / g * dT * np.diff(phlev)) + eff_Cp_s * dT_s

    return term_diff


def latent_heat_difference(h2o_2, h2o_1):
    """
    Calculate the difference in energy from latent heating between two
    water vapour profiles (2 - 1).

    Parameters:
        h2o_2 (ndarray): water vapour content [kg m^-2]
        h2o_1 (ndarray): water vapour content [kg m^-2]
    Returns:
        float: energy difference [J m^-2]
    """

    Lv = constants.Lv  # TODO: include pressure/temperature dependence?
    term_diff = np.sum((h2o_2 - h2o_1) * Lv)

    return term_diff


def interp_variable(variable, convective_heating, lim):
    """
    Find the value of a variable corresponding to where the convective
    heating equals a certain specified value (lim).

    Parameters:
        variable (ndarray): variable to be interpolated
        convective_heating (ndarray): interpolate based on where this variable
            equals 'lim'
        lim (float/int): value of 'convective_heating' used to find the
            corresponding value of 'variable'

    Returns:
         float: interpolated value of 'variable'
    """
    positive_i = int(np.argmax(convective_heating > lim))
    contop_index = int(np.argmax(convective_heating[positive_i:] < lim)) + positive_i

    # Create auxiliary arrays storing the Qr, T and p values above and below
    # the threshold value. These arrays are used as input for the interpolation
    # in the next step.
    heat_array = np.array(
        [convective_heating[contop_index - 1], convective_heating[contop_index]]
    )
    var_array = np.array([variable[contop_index - 1], variable[contop_index]])

    # Interpolate the values to where the convective heating rate equals `lim`.
    return interp1d(heat_array, var_array)(lim)


class Convection(Component, metaclass=abc.ABCMeta):
    """Base class to define abstract methods for convection schemes."""

    @abc.abstractmethod
    def stabilize(self, atmosphere, lapse, surface, timestep):
        """Stabilize the temperature profile by redistributing energy.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
            lapse (konrad.lapsereate.LapseRate): Callable `f(p, T)` that
                returns a temperature lapse rate in [K/day].
            surface (konrad.surface): Surface model.
            timestep (float): Timestep width [day].
        """


class NonConvective(Convection):
    """Do not apply convection."""

    def stabilize(self, *args, **kwargs):
        pass


class HardAdjustment(Convection):
    """Instantaneous adjustment of temperature profiles"""

    def __init__(self, entrainment=None, etol=1e-4):
        """Initialize the convective adjustment.

        Parameters:
            entrainment (:py:class:`konrad.entrainment.Entrainment`):
                Optional entrainment component.
            etol (float): Threshold for the allowed energy difference between
                the input temperature and the stabilized temperature profile.
                The threshold is internally scaled using the surface heat capacity,
                therefore its unit is more relative than physical.
        """
        if entrainment is None:
            self._entrainment = NoEntrainment()
        else:
            self._entrainment = entrainment

        self.etol = etol

    @property
    def netcdf_subgroups(self):
        return {"entrainment": self._entrainment}

    def stabilize(self, atmosphere, lapse, surface, timestep):
        T_rad = atmosphere["T"][0, :]
        p = atmosphere["plev"]

        # Find convectively adjusted temperature profile.
        T_new, T_s_new = self.convective_adjustment(
            atmosphere=atmosphere,
            lapse=lapse,
            surface=surface,
            timestep=timestep,
        )
        # get convective top temperature and pressure
        self.update_convective_top(T_rad, T_new, p, timestep=timestep)
        # Update atmospheric temperatures as well as surface temperature.
        atmosphere["T"][0, :] = T_new
        surface["temperature"][0] = T_s_new

    def convective_adjustment(self, atmosphere, lapse, surface, timestep=0.1):
        """
        Find the energy-conserving temperature profile using upper and lower
        bound profiles (calculated from surface temperature extremes: no change
        for upper bound and coldest atmospheric temperature for lower bound)
        and an iterative procedure between them.
        Return the atmospheric temperature profile which satisfies energy
        conservation.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
            lapse (konrad.lapsereate.LapseRate): Callable `f(p, T)` that
                returns a temperature lapse rate in [K/day].
            surface (konrad.surface):
                surface associated with old temperature profile
            timestep (float): only required for slow convection [days]

        Returns:
            ndarray: atmospheric temperature profile [K]
            float: surface temperature [K]
        """
        # This is the temperature profile required if we have a set-up with a
        # fixed surface temperature. In this case, energy is not conserved.
        if isinstance(surface, FixedTemperature):
            T_con = self.convective_profile(
                atmosphere, surface["temperature"], lapse, timestep=timestep
            )
            return T_con, surface["temperature"]

        # Otherwise we should conserve energy --> our energy change should be
        # less than the threshold 'near_zero'.
        # The threshold is scaled with the effective heat capacity of the
        # surface, ensuring that very thick surfaces reach the target.
        near_zero = float(surface.heat_capacity) * self.etol

        # Find the energy difference if there is no change to surface temp due
        # to convective adjustment. In this case the new profile should be
        # associated with an increase in energy in the atmosphere.
        surfaceTpos = surface["temperature"]
        T_con, diff_pos = self.create_and_check_profile(
            atmosphere, surface, surfaceTpos, lapse, timestep=timestep
        )

        # For other cases, if we find a decrease or approx no change in energy,
        # the atmosphere is not being warmed by the convection,
        # as it is not unstable to convection, so no adjustment is applied.
        if diff_pos < near_zero:
            return T_con, surface["temperature"]

        # If the atmosphere is unstable to convection, a fixed surface
        # temperature produces an increase in energy, as convection warms the
        # atmosphere. Therefore 'surfaceTpos' is an upper bound for the
        # energy-conserving surface temperature we are trying to find.
        # Taking the surface temperature as the coldest temperature in the
        # radiative profile gives us a lower bound. In this case, convection
        # would not warm the atmosphere, so we do not change the atmospheric
        # temperature profile and calculate the energy change simply from the
        # surface temperature change.
        surfaceTneg = atmosphere["T"][0, 0]
        eff_Cp_s = surface.heat_capacity
        diff_neg = eff_Cp_s * (surfaceTneg - surface["temperature"])
        if np.abs(diff_neg) < near_zero:
            return T_con, surfaceTneg

        # Now we have a upper and lower bound for the surface temperature of
        # the energy conserving profile. Iterate to get closer to the energy-
        # conserving temperature profile.
        counter = 0
        while diff_pos >= near_zero and np.abs(diff_neg) >= near_zero:
            # Use a surface temperature between our upper and lower bounds and
            # closer to the bound associated with a smaller energy change.
            surfaceT = surfaceTneg + (surfaceTpos - surfaceTneg) * (-diff_neg) / (
                -diff_neg + diff_pos
            )
            # Calculate temperature profile and energy change associated with
            # this surface temperature.
            T_con, diff = self.create_and_check_profile(
                atmosphere, surface, surfaceT, lapse, timestep=timestep
            )

            # Update either upper or lower bound.
            if diff > 0:
                diff_pos = diff
                surfaceTpos = surfaceT
            else:
                diff_neg = diff
                surfaceTneg = surfaceT

            # to avoid getting stuck in a loop if something weird is going on
            counter += 1
            if counter == 100:
                raise ValueError("No energy conserving convective profile can be found")

        return T_con, surfaceT

    @staticmethod
    def get_moist_adiabat(atmosphere, surfaceT, lapse, **kwargs):
        """
        Assuming a particular surface temperature (surfaceT), create a new
        profile, following the specified lapse rate (lp) for the region where
        the convectively adjusted atmosphere is warmer than the radiative one.
        Above this, use the radiative profile, as convection is not allowed in
        the stratosphere.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
            surfaceT (float): surface temperature [K]
            lapse (konrad.lapsereate.LapseRate): Callable `f(p, T)` that
                returns a temperature lapse rate in [K/day].

        Returns:
             ndarray: convectively adjusted temperature profile [K]
        """
        # Kudos to Jiawei Bao for this neat implementation of the dTdp integration!
        p = atmosphere["plev"]
        ph = atmosphere["phlev"]
        Ts = surfaceT

        r = ode(lapse).set_integrator("lsoda", atol=1e-4)
        r.set_initial_value(Ts, ph[0])

        T = np.zeros_like(p)
        i = 0
        dp = np.hstack((np.array([p[0] - ph[0]]), np.diff(p)))
        while r.successful() and r.t > p.min():
            r.integrate(r.t + dp[i])
            T[i] = r.y[0]
            i += 1

        return T

    def convective_profile(self, atmosphere, surfaceT, lapse, **kwargs):
        # The convective adjustment is only applied to the atmospheric profile,
        # if it causes heating somewhere
        T_rad = atmosphere["T"][-1]
        T_con = self.get_moist_adiabat(atmosphere, surfaceT, lapse)

        # Entrain dry air into the convective atmosphere column.
        T_con = self._entrainment.entrain(T_con, atmosphere)

        # Combine radiative and convective temperature profiles.
        if np.any(T_con > T_rad):
            contop = np.max(np.where(T_con > T_rad))
            T_con[contop + 1 :] = T_rad[contop + 1 :]
        else:
            return T_rad

        return T_con

    def create_and_check_profile(
        self, atmosphere, surface, surfaceT, lapse, timestep=0.1
    ):
        """Create a convectively adjusted temperature profile and calculate how
        close it is to satisfying energy conservation.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
            surface (konrad.surface):
                surface associated with old temperature profile
            surfaceT (float): surface temperature of the new profile
            lp (ndarray): lapse rate in K/Pa
            timestep (float): not required in this case

        Returns:
            ndarray: new atmospheric temperature profile
            float: energy difference between the new profile and the old one
        """
        T_rad = atmosphere["T"][-1]
        T_con = self.convective_profile(atmosphere, surfaceT, lapse, timestep=timestep)

        eff_Cp_s = surface.heat_capacity

        diff = energy_difference(
            T_con,
            T_rad,
            surfaceT,
            surface["temperature"],
            atmosphere["phlev"],
            eff_Cp_s,
        )
        return T_con, float(diff)

    def update_convective_top(self, T_rad, T_con, p, timestep=0.1, lim=0.2):
        """
        Find the pressure and temperature where the radiative heating has a
        certain value.

        Note:
            In the HardAdjustment case, for a contop temperature that is not
            dependent on the number or distribution of pressure levels, it is
            better to take a value of lim not equal or very close to zero.

        Parameters:
            T_rad (ndarray): radiative temperature profile [K]
            T_con (ndarray): convectively adjusted temperature profile [K]
            p (ndarray): model pressure levels [Pa]
            timestep (float): model timestep [days]
            lim (float): Threshold value [K/day].
        """
        convective_heating = (T_con - T_rad) / timestep
        self.create_variable("convective_heating_rate", convective_heating)

        if np.any(convective_heating > lim):  # if there is convective heating
            # find the values of pressure and temperature at the convective top,
            # as defined by a threshold convective heating value
            contop_p = interp_variable(p, convective_heating, lim)
            contop_T = interp_variable(T_con, convective_heating, lim)

            # At every level above the contop, the convective heating is either
            # zero (no convection is applied, HardAdj) or negative (RlxAdj).
            # Convection acts to warm the upper troposphere, but it may either
            # warm (normally) or cool (at certain times during the diurnal
            # cycle) the lower troposphere. Therefore, we search for the
            # convective top index from the top going downwards.
            contop_index = len(convective_heating) - np.argmin(
                convective_heating[::-1] <= 0
            )

        else:  # if there is no convective heating
            contop_index, contop_p, contop_T = np.nan, np.nan, np.nan

        for name, value in [
            ("convective_top_plev", contop_p),
            ("convective_top_temperature", contop_T),
            ("convective_top_index", contop_index),
        ]:
            self.create_variable(name, np.array([value]))

        return

    def update_convective_top_height(self, z, lim=0.2):
        """Find the height where the radiative heating has a certain value.

        Parameters:
            z (ndarray): height array [m]
            lim (float): Threshold convective heating value [K/day]
        """
        convective_heating = self.get("convective_heating_rate")[0]
        if np.any(convective_heating > lim):  # if there is convective heating
            contop_z = interp_variable(z, convective_heating, lim=lim)
        else:  # if there is no convective heating
            contop_z = np.nan
        self.create_variable("convective_top_height", np.array([contop_z]))
        return


class FixedDynamicalHeating(HardAdjustment):
    """Adjustment with a fixed convective (dynamical) heating rate."""

    def __init__(self, heating=None, *args, **kwargs):
        """
        Parameters:
            heating (ndarray): Array of convective heating values [K/day]
        """
        super().__init__(*args, **kwargs)
        self._heating = heating

    def stabilize(self, atmosphere, lapse, surface, timestep):
        p = atmosphere["plev"]
        T_rad = atmosphere["T"][-1]
        T_new = T_rad + timestep * self._heating

        self.update_convective_top(T_rad, T_new, p, timestep=timestep)

        atmosphere["T"][-1] = T_new
