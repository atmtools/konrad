# -*- coding: utf-8 -*-
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

from konrad import constants
from konrad.component import Component
from konrad.surface import FixedTemperature


__all__ = [
    'energy_difference',
    'latent_heat_difference',
    'interp_variable',
    'pressure_lapse_rate',
    'Convection',
    'NonConvective',
    'HardAdjustment',
    'RelaxedAdjustment',
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

    term_diff = - np.sum(Cp/g * dT * np.diff(phlev)) + eff_Cp_s * dT_s

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
    term_diff = np.sum((h2o_2-h2o_1) * Lv)

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
    contop_index = int(np.argmax(
        convective_heating[positive_i:] < lim)) + positive_i

    # Create auxiliary arrays storing the Qr, T and p values above and below
    # the threshold value. These arrays are used as input for the interpolation
    # in the next step.
    heat_array = np.array([convective_heating[contop_index - 1],
                           convective_heating[contop_index]])
    var_array = np.array([variable[contop_index - 1], variable[contop_index]])

    # Interpolate the values to where the convective heating rate equals `lim`.
    return interp1d(heat_array, var_array)(lim)


def pressure_lapse_rate(p, phlev, T, lapse):
    """
    Calculate the pressure lapse rate (change in temperature with pressure)
    from the height lapse rate (change in temperature with height).

    Parameters:
        p (ndarray): pressure levels
        phlev (ndarray): pressure half-levels
        T (ndarray): temperature profile
        lapse (ndarray): lapse rate [K/m] defined on pressure half-levels
    Returns:
        ndarray: pressure lapse rate [K/Pa]
    """
    density_p = typhon.physics.density(p, T)
    # Interpolate density onto pressure half-levels
    density = interp1d(p, density_p, fill_value='extrapolate')(phlev[:-1])

    g = constants.earth_standard_gravity
    lp = -lapse / (g * density)
    return lp


class Convection(Component, metaclass=abc.ABCMeta):
    """Base class to define abstract methods for convection schemes."""
    @abc.abstractmethod
    def stabilize(self, atmosphere, lapse, surface, timestep):
        """Stabilize the temperature profile by redistributing energy.

        Parameters:
              atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
              lapse (ndarray): Temperature lapse rate [K/day].
              surface (konrad.surface): Surface model.
              timestep (float): Timestep width [day].
        """


class NonConvective(Convection):
    """Do not apply convection."""
    def stabilize(self, *args, **kwargs):
        pass


class HardAdjustment(Convection):
    """Instantaneous adjustment of temperature profiles"""
    def stabilize(self, atmosphere, lapse, surface, timestep):

        T_rad = atmosphere['T'][0, :]
        p = atmosphere['plev']

        # Find convectively adjusted temperature profile.
        T_new, T_s_new = self.convective_adjustment(
            p=p,
            phlev=atmosphere['phlev'],
            T_rad=T_rad,
            lapse=lapse,
            surface=surface,
            timestep=timestep,
        )
        # get convective top temperature and pressure
        self.update_convective_top(T_rad, T_new, p, timestep=timestep)
        # Update atmospheric temperatures as well as surface temperature.
        atmosphere['T'][0, :] = T_new
        surface['temperature'][0] = T_s_new

    def convective_adjustment(self, p, phlev, T_rad, lapse, surface,
                              timestep=0.1):
        """
        Find the energy-conserving temperature profile using upper and lower
        bound profiles (calculated from surface temperature extremes: no change
        for upper bound and coldest atmospheric temperature for lower bound)
        and an iterative procedure between them.
        Return the atmospheric temperature profile which satisfies energy
        conservation.

        Parameters:
            p (ndarray): pressure levels [Pa]
            phlev (ndarray): half pressure levels [Pa]
            T_rad (ndarray): old atmospheric temperature profile [K]
            lapse (ndarray): critical lapse rate [K/m] defined on pressure
                half-levels
            surface (konrad.surface):
                surface associated with old temperature profile
            timestep (float): only required for slow convection [days]

        Returns:
            ndarray: atmospheric temperature profile [K]
            float: surface temperature [K]
        """
        lp = pressure_lapse_rate(p, phlev, T_rad, lapse)

        # This is the temperature profile required if we have a set-up with a
        # fixed surface temperature. In this case, energy is not conserved.
        if isinstance(surface, FixedTemperature):
            T_con = self.convective_profile(
                T_rad, p, phlev, surface['temperature'], lp, timestep=timestep)
            return T_con, surface['temperature']

        # Otherwise we should conserve energy --> our energy change should be
        # less than the threshold 'near_zero'.
        # The threshold is scaled with the effective heat capacity of the
        # surface, ensuring that very thick surfaces reach the target.
        near_zero = float(surface.heat_capacity / 1e13)

        # Find the energy difference if there is no change to surface temp due
        # to convective adjustment. In this case the new profile should be
        # associated with an increase in energy in the atmosphere.
        surfaceTpos = surface['temperature']
        T_con, diff_pos = self.create_and_check_profile(
            T_rad, p, phlev, surface, surfaceTpos, lp, timestep=timestep)

        # For other cases, if we find a decrease or approx no change in energy,
        # the atmosphere is not being warmed by the convection,
        # as it is not unstable to convection, so no adjustment is applied.
        if diff_pos < near_zero:
            return T_con, surface['temperature']

        # If the atmosphere is unstable to convection, a fixed surface
        # temperature produces an increase in energy, as convection warms the
        # atmosphere. Therefore 'surfaceTpos' is an upper bound for the
        # energy-conserving surface temperature we are trying to find.
        # Taking the surface temperature as the coldest temperature in the
        # radiative profile gives us a lower bound. In this case, convection
        # would not warm the atmosphere, so we do not change the atmospheric
        # temperature profile and calculate the energy change simply from the
        # surface temperature change.
        surfaceTneg = np.array([np.min(T_rad)])
        eff_Cp_s = surface.heat_capacity
        diff_neg = eff_Cp_s * (surfaceTneg - surface['temperature'])
        if np.abs(diff_neg) < near_zero:
            return T_con, surfaceTneg

        # Now we have a upper and lower bound for the surface temperature of
        # the energy conserving profile. Iterate to get closer to the energy-
        # conserving temperature profile.
        counter = 0
        while diff_pos >= near_zero and np.abs(diff_neg) >= near_zero:
            # Use a surface temperature between our upper and lower bounds and
            # closer to the bound associated with a smaller energy change.
            surfaceT = (surfaceTneg + (surfaceTpos - surfaceTneg)
                        * (-diff_neg) / (-diff_neg + diff_pos))
            # Calculate temperature profile and energy change associated with
            # this surface temperature.
            T_con, diff = self.create_and_check_profile(
                T_rad, p, phlev, surface, surfaceT, lp, timestep=timestep)

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
                raise ValueError(
                    "No energy conserving convective profile can be found"
                )

        return T_con, surfaceT

    def convective_profile(self, T_rad, p, phlev, surfaceT, lp, **kwargs):
        """
        Assuming a particular surface temperature (surfaceT), create a new
        profile, following the specified lapse rate (lp) for the region where
        the convectively adjusted atmosphere is warmer than the radiative one.
        Above this, use the radiative profile, as convection is not allowed in
        the stratosphere.

        Parameters:
            T_rad (ndarray): radiative temperature profile [K]
            p (ndarray): pressure levels [Pa]
            phlev (ndarray): pressure half-levels [Pa]
            surfaceT (float): surface temperature [K]
            lp (ndarray): pressure lapse rate [K/Pa]

        Returns:
             ndarray: convectively adjusted temperature profile [K]
        """
        # for the lapse rate integral use a different dp, considering that the
        # lapse rate is given on half levels
        dp_lapse = np.hstack((np.array([p[0] - phlev[0]]), np.diff(p)))
        T_con = surfaceT - np.cumsum(dp_lapse * lp)

        if np.any(T_con > T_rad):
            contop = np.max(np.where(T_con > T_rad))
            T_con[contop+1:] = T_rad[contop+1:]
        else:
            # convective adjustment is only applied to the atmospheric profile,
            # if it causes heating somewhere
            T_con = T_rad

        return T_con

    def create_and_check_profile(self, T_rad, p, phlev, surface, surfaceT, lp,
                                 timestep=0.1):
        """Create a convectively adjusted temperature profile and calculate how
        close it is to satisfying energy conservation.

        Parameters:
            T_rad (ndarray): old atmospheric temperature profile
            p (ndarray): pressure levels
            phlev (ndarray): half pressure levels
            surface (konrad.surface):
                surface associated with old temperature profile
            surfaceT (float): surface temperature of the new profile
            lp (ndarray): lapse rate in K/Pa
            timestep (float): not required in this case

        Returns:
            ndarray: new atmospheric temperature profile
            float: energy difference between the new profile and the old one
        """
        T_con = self.convective_profile(T_rad, p, phlev, surfaceT, lp,
                                        timestep=timestep)

        eff_Cp_s = surface.heat_capacity

        diff = energy_difference(T_con, T_rad, surfaceT,
                                 surface['temperature'], phlev, eff_Cp_s)
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
        self.create_variable('convective_heating_rate', convective_heating)

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
            contop_index = (len(convective_heating) -
                            np.argmin(convective_heating[::-1] <= 0))

        else:  # if there is no convective heating
            contop_index, contop_p, contop_T = np.nan, np.nan, np.nan

        for name, value in [('convective_top_plev', contop_p),
                            ('convective_top_temperature', contop_T),
                            ('convective_top_index', contop_index),
                            ]:
            self.create_variable(name, np.array([value]))

        return

    def update_convective_top_height(self, z, lim=0.2):
        """Find the height where the radiative heating has a certain value.

        Parameters:
            z (ndarray): height array [m]
            lim (float): Threshold convective heating value [K/day]
        """
        convective_heating = self.get('convective_heating_rate')[0]
        if np.any(convective_heating > lim):  # if there is convective heating
            contop_z = interp_variable(z, convective_heating, lim=lim)
        else:  # if there is no convective heating
            contop_z = np.nan
        self.create_variable('convective_top_height', np.array([contop_z]))
        return


class RelaxedAdjustment(HardAdjustment):
    """Adjustment with relaxed convection in upper atmosphere.

    This convection scheme allows for a transition regime between a
    convectively driven troposphere and the radiatively balanced stratosphere.
    """
    def __init__(self, tau=None):
        """
        Parameters:
            tau (ndarray): Array of convective timescale values [days]
        """
        self.convective_tau = tau

    def get_convective_tau(self, p):
        """Return a convective timescale profile.

        Parameters:
            p (ndarray): Pressure levels [Pa].

        Returns:
            ndarray: Convective timescale profile [days].
        """
        if self.convective_tau is not None:
            return self.convective_tau

        tau0 = 1/24  # 1 hour
        tau = tau0*np.exp(p[0] / p)

        return tau

    def convective_profile(self, T_rad, p, phlev, surfaceT, lp, timestep):
        """
        Assuming a particular surface temperature (surfaceT), create a new
        profile, which tries to follow the specified lapse rate (lp). How close
        it gets to following the specified lapse rate depends on the convective
        timescale and model timestep.

        Parameters:
            T_rad (ndarray): radiative temperature profile [K]
            p (ndarray): pressure levels [Pa]
            phlev (ndarray): pressure half-levels [Pa]
            surfaceT (float): surface temperature [K]
            lp (ndarray): pressure lapse rate [K/Pa]
            timestep (float/int): model timestep [days]

        Returns:
             ndarray: convectively adjusted temperature profile [K]
        """
        # For the lapse rate integral use a dp, which takes into account that
        # the lapse rate is given on the model half-levels.
        dp_lapse = np.hstack((np.array([p[0] - phlev[0]]), np.diff(p)))

        tau = self.get_convective_tau(p)

        tf = 1 - np.exp(-timestep / tau)
        T_con = T_rad * (1 - tf) + tf * (surfaceT - np.cumsum(dp_lapse * lp))

        return T_con
