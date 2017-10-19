# -*- coding: utf-8 -*-
"""This module contains classes handling different convection schemes."""
import abc

import numpy as np
import typhon

from conrad import (constants, utils)
from conrad.surface import SurfaceFixedTemperature


__all__ = [
    'Convection',
    'NonConvective',
    'HardAdjustment',
    'RelaxedAdjustment',
]


def energy_difference(T_2, T_1, sst_2, sst_1, dp, eff_Cp_s):
    """
    Calculate the energy difference between two atmospheric profiles (2 - 1).

    Parameters:
        T_2: atmospheric temperature profile (2)
        T_1: atmospheric temperature profile (1)
        sst_2: surface temperature (2)
        sst_1: surface temperature (1)
        dp: pressure thicknesses of levels,
            must be the same for both atmospheric profiles
        eff_Cp_s: effective heat capacity of surface
    """
    Cp = constants.isobaric_mass_heat_capacity
    g = constants.g

    dT = T_2 - T_1  # convective temperature change of atmosphere
    dT_s = sst_2 - sst_1  # of surface
    termdiff = - np.sum(Cp/g * dT * dp) + eff_Cp_s * dT_s

    return termdiff


class Convection(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for convection schemes."""
    def __init__(self):
        """Create a convection scheme.

        Parameters:
        """
        pass

    @abc.abstractmethod
    def stabilize(self, **kwargs):
        """Stabilize the temperature profile by redistributing energy.

        Parameters:

        Returns:
            ndarray: Stabilized temperature profile [K].
        """


class NonConvective(Convection):
    """Do not apply convection."""
    def stabilize(self, p, phlev, T_rad, lapse, surface, timestep=0.1):
        return T_rad, surface.temperature[0]


class HardAdjustment(Convection):
    """Instantanuous adjustment of temperature profiles"""
    def stabilize(self, p, phlev, T_rad, lapse, surface, timestep=0.1):
        """
        Find the energy-conserving temperature profile using a iterative
        procedure with test profiles. Update the atmospheric temperature
        profile to this one.

        Parameters:
            timestep: float, only required for slow convection
        """
        near_zero = 0.00001
        density1 = typhon.physics.density(p, T_rad)
        # TODO (Sally): I dont think that the following line is on purpose.
        density = utils.calculate_halflevel_pressure(density1)

        g = constants.g
        # TODO: Find a clean way to handle different lapse rate versions.
        lp = -lapse[:] / (g*density)

        # find energy difference if there is no change to surface temp due to
        # convective adjustment. in this case the new profile should be
        # associated with an increase in energy in the atmosphere.
        surfaceTpos = surface.temperature.values
        T_con, diffpos = self.test_profile(T_rad, p, phlev, surface,
                                           surfaceTpos, lp,
                                           timestep=timestep)
        # this is the temperature profile required if we have a set-up with a
        # fixed surface temperature, then the energy does not matter.
        if isinstance(surface, SurfaceFixedTemperature):
            return T_con, surface.temperature
        # for other cases, if we find a decrease or approx no change in energy,
        # the atmosphere is not being warmed by the convection,
        # as it is not unstable to convection, so no adjustment is applied
        if diffpos < near_zero:
            return T_con, surface.temperature

        # if the atmosphere is unstable to convection, a fixed surface temp
        # produces an increase in energy (as convection warms the atmosphere).
        # this surface temperature is an upper bound to the energy-conserving
        # surface temperature.
        # now we reduce surface temperature until we find an adjusted profile
        # that is associated with an energy loss.
        surfaceTneg = surfaceTpos - 1
        T_con, diffneg = self.test_profile(T_rad, p, phlev, surface,
                                           surfaceTneg, lp,
                                           timestep=timestep)
        # good guess for energy-conserving profile
        if np.abs(diffneg) < near_zero:
            return T_con, surfaceTneg
        # if surfaceTneg = surfaceTpos - 1 is not negative enough to produce an
        # energy loss, keep reducing surfaceTneg to find a lower bound for the
        # uncertainty range of the energy-conserving surface temperature
        while diffneg > 0:
            diffpos = diffneg
            # update surfaceTpos to narrow the uncertainty range
            surfaceTpos = surfaceTneg
            surfaceTneg -= 1
            T_con, diffneg = self.test_profile(T_rad, p, phlev, surface,
                                               surfaceTneg, lp,
                                               timestep=timestep)
            # again for the case that this surface temperature happens to
            # be a good guess (sufficiently close to energy conserving)
            if np.abs(diffneg) < near_zero:
                return T_con, surfaceTneg

        # Now we have a upper and lower bound for the surface temperature of
        # the energy conserving profile. Iterate to get closer to the energy-
        # conserving temperature profile.
        counter = 0
        while diffpos >= near_zero and -diffneg >= near_zero:
            surfaceT = (surfaceTneg + (surfaceTpos - surfaceTneg)
                        * (-diffneg) / (-diffneg + diffpos))
            T_con, diff = self.test_profile(T_rad, p, phlev, surface, surfaceT,
                                            lp, timestep=timestep)
            if diff > 0:
                diffpos = diff
                surfaceTpos = surfaceT
            else:
                diffneg = diff
                surfaceTneg = surfaceT
            # to avoid getting stuck in a loop if something weird is going on
            counter += 1
            if counter == 100:
                raise ValueError(
                    "No energy conserving convective profile can be found"
                )

        # save new temperature profile
        return T_con, surfaceT

    def test_profile(self, T_rad, p, phlev, surface, surfaceT, lp,
                     timestep=0.1):
        """
        Assuming a particular surface temperature (surfaceT), create a new
        profile, following the specified lapse rate (lp) for the region where
        the convectively adjusted atmosphere is warmer than the radiative one.

        Parameters:
            surfaceT: float, surface temperature of the new profile
            lp: lapse rate in K/Pa
            timestep: float, not required in this case

        Returns:
            ndarray: new atmospheric temperature profile
            float: energy difference between the new profile and the old one
        """
        # dp, thicknesses of atmosphere layers, for energy calculation
        dp = np.diff(phlev)
        # for lapse rate integral
        dp_lapse = np.hstack((np.array([p[0] - phlev[0]]), np.diff(p)))

        T_con = surfaceT - np.cumsum(dp_lapse * lp[:-1])
        if np.any(T_con > T_rad):
            contop = np.max(np.where(T_con > T_rad))
            T_con[contop+1:] = T_rad[contop+1:]
        else:
            T_con = T_rad
            contop = 0

        eff_Cp_s = surface.rho * surface.cp * surface.dz

        diff = energy_difference(T_con, T_rad, surfaceT, surface.temperature,
                                 dp, eff_Cp_s)

        return T_con, float(diff)


class RelaxedAdjustment(HardAdjustment):
    """Adjustment with relaxed convection in upper atmosphere.

    This convection scheme allows for a transition regime between a
    convectively driven troposphere and the radiatively balanced stratosphere.
    """
    def __init__(self, *args, tau=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.convective_tau = tau

    def test_profile(self, T_rad, p, phlev, surface, surfaceT, lp,
                     timestep=0.1):
        """
        Assuming a particular surface temperature (surfaceT), create a new
        profile, using the convective timescale and specified lapse rate (lp).

        Parameters:
            surfaceT: float, surface temperature of the new profile
            lp: lapse rate in K/Pa
            timestep: float, timestep of simulation

        Returns:
            ndarray: new atmospheric temperature profile
            float: energy difference between the new profile and the old one
        """
        dp = np.diff(phlev)
        dp_lapse = np.hstack((np.array([p[0] - phlev[0]]), np.diff(p)))

        tf = 1 - np.exp(-timestep / self.convective_tau)
        T_con = T_rad * (1 - tf) + tf * (surfaceT - np.cumsum(dp_lapse * lp[:-1]))
        T_con = T_con

        eff_Cp_s = surface.rho * surface.cp * surface.dz

        diff = energy_difference(T_con, T_rad, surfaceT, surface.temperature,
                                 dp, eff_Cp_s)

        return T_con, float(diff.values)


