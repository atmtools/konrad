# -*- coding: utf-8 -*-
import logging

import typhon
import numpy as np
from xarray import DataArray


import conrad
from conrad import constants
from conrad import utils
from conrad.atmosphere.abc import Atmosphere


__all__ = [
    'AtmosphereConvective',
    'AtmosphereSlowConvective',
    'AtmosphereMoistConvective',
    'AtmosphereSlowMoistConvective',
    'AtmosphereConUp',
    'AtmosphereConvectiveFlux',
]

logger = logging.getLogger()


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


class AtmosphereConvective(Atmosphere):
    """Atmosphere model with preserved RH and fixed temperature lapse rate.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio. In addition, a simple
    convection parameterization is used.
    """
    def __init__(self, *args, lapse=0.0065, **kwargs):

        super().__init__(*args, **kwargs)
        if isinstance(lapse, float):
            # make an array of lapse rate values, corresponding to the half
            # pressure levels
            lapse_array = lapse * np.ones((1, self['phlev'].size))
            self['lapse'] = DataArray(lapse_array, dims=('time', 'phlev'))
        elif isinstance(lapse, np.ndarray):
            # Here the input lapse rate is given on the full pressure levels,
            # we need to convert it, so that it is on the half levels.
            lapse_phlev = utils.calculate_halflevel_pressure(lapse.values)
            self['lapse'] = DataArray(lapse_phlev, dims=('time', 'phlev'))

        utils.append_description(self)  # Append variable descriptions.

    def save_profile(self, surface, T_con, surfaceT):
        """
        Update the surface and atmospheric temperatures to surfaceT and T_con.

        Parameters:
            surfaceT: float
            T_con: ndarray
        """
        surface['temperature'][0] = surfaceT
        self['T'].values = T_con[np.newaxis, :]

    def convective_adjustment(self, surface, timestep=0.1):
        """
        Find the energy-conserving temperature profile using a iterative
        procedure with test profiles. Update the atmospheric temperature
        profile to this one.

        Parameters:
            timestep: float, only required for slow convection
        """
        near_zero = 0.00001
        T_rad = self['T'][0, :]
        p = self['plev']
        lapse = self.lapse[0, :]
        density1 = typhon.physics.density(p, T_rad)
        density = utils.calculate_halflevel_pressure(density1.values)

        g = constants.g
        lp = -lapse[:].values / (g*density)

        # find energy difference if there is no change to surface temp due to
        # convective adjustment. in this case the new profile should be
        # associated with an increase in energy in the atmosphere.
        surfaceTpos = surface.temperature.values
        T_con, diffpos = self.test_profile(surface, surfaceTpos, lp,
                                           timestep=timestep)
        # this is the temperature profile required if we have a set-up with a
        # fixed surface temperature, then the energy does not matter.
        if isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            self['T'].values = T_con.values[np.newaxis, :]
        # for other cases, if we find a decrease or approx no change in energy,
        # the atmosphere is not being warmed by the convection,
        # as it is not unstable to convection, so no adjustment is applied
        if diffpos < near_zero:
            return None

        # if the atmosphere is unstable to convection, a fixed surface temp
        # produces an increase in energy (as convection warms the atmosphere).
        # this surface temperature is an upper bound to the energy-conserving
        # surface temperature.
        # now we reduce surface temperature until we find an adjusted profile
        # that is associated with an energy loss.
        surfaceTneg = surfaceTpos - 1
        T_con, diffneg = self.test_profile(surface, surfaceTneg, lp,
                                           timestep=timestep)
        # good guess for energy-conserving profile
        if np.abs(diffneg) < near_zero:
            self.save_profile(surface, T_con, surfaceTneg)
            return None
        # if surfaceTneg = surfaceTpos - 1 is not negative enough to produce an
        # energy loss, keep reducing surfaceTneg to find a lower bound for the
        # uncertainty range of the energy-conserving surface temperature
        while diffneg > 0:
            diffpos = diffneg
            # update surfaceTpos to narrow the uncertainty range
            surfaceTpos = surfaceTneg
            surfaceTneg -= 1
            T_con, diffneg = self.test_profile(surface, surfaceTneg, lp,
                                               timestep=timestep)
            # again for the case that this surface temperature happens to
            # be a good guess (sufficiently close to energy conserving)
            if np.abs(diffneg) < near_zero:
                self.save_profile(surface, T_con, surfaceTneg)
                return None

        # Now we have a upper and lower bound for the surface temperature of
        # the energy conserving profile. Iterate to get closer to the energy-
        # conserving temperature profile.
        counter = 0
        while diffpos >= near_zero and -diffneg >= near_zero:
            surfaceT = (surfaceTneg + (surfaceTpos - surfaceTneg)
                        * (-diffneg) / (-diffneg + diffpos))
            T_con, diff = self.test_profile(surface, surfaceT, lp,
                                            timestep=timestep)
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
        self.save_profile(surface, T_con, surfaceT)

    def test_profile(self, surface, surfaceT, lp, timestep=0.1):
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
        T_rad = self['T'][0, :]
        p = self['plev']
        phlev = self['phlev']
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
        self['convective_top'] = DataArray([contop], dims=('time',))

        eff_Cp_s = surface.rho * surface.cp * surface.dz

        diff = energy_difference(T_con, T_rad, surfaceT, surface.temperature,
                                 dp, eff_Cp_s)

        return T_con, float(diff)

    def adjust(self, heatingrates, timestep, surface):

        # Apply heatingrates to temperature profile.
        self['T'] += heatingrates * timestep

        # Apply convective adjustment
        self.convective_adjustment(surface)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()


class AtmosphereSlowConvective(AtmosphereConvective):
    """
    Atmosphere with a time dependent convective adjustment.

    Here the convective adjustment occurs throughout the whole atmosphere, but
    tau (the convective timescale) should be chosen to be very large in the
    middle and upper atmosphere.
    """
    def __init__(self, *args, tau=0, **kwargs):
        super().__init__(*args, **kwargs)

        self['convective_tau'] = DataArray(tau[np.newaxis, :],
                                           dims=('time', 'plev'))

        utils.append_description(self)

    def test_profile(self, surface, surfaceT, lp, timestep):
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
        T_rad = self['T'][0, :]
        p = self['plev']
        phlev = self['phlev']
        dp = np.diff(phlev)
        dp_lapse = np.hstack((np.array([p[0] - phlev[0]]), np.diff(p)))

        tau = self['convective_tau'][0]
        tf = 1 - np.exp(-timestep/tau)
        T_con = T_rad*(1 - tf) + tf*(surfaceT - np.cumsum(dp_lapse * lp[:-1]))
        T_con = T_con.values

        eff_Cp_s = surface.rho * surface.cp * surface.dz

        diff = energy_difference(T_con, T_rad, surfaceT, surface.temperature,
                                 dp, eff_Cp_s)

        return T_con, float(diff.values)

    def adjust(self, heatingrates, timestep, surface):

        # Apply heatingrates to temperature profile.
        self['T'] += heatingrates * timestep

        # Apply convective adjustment
        self.convective_adjustment(surface, timestep=timestep)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()


class AtmosphereMoistConvective(AtmosphereConvective):
    """Atmosphere model with preserved RH and a temperature and humidity
    -dependent lapse rate.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio. In addition, a convection
    parameterization is used, which sets the lapse rate to the moist adiabat,
    calculated from the previous temperature and humidity profiles.
    """
    def moistlapse(self):
        """Updates the atmospheric lapse rate for the convective adjustment
        according to the moist adiabat, which is calculated from the
        atmospheric temperature and humidity profiles. The lapse rate is in
        units of K/km.
        """
        g = constants.g
        Lv = 2501000
        R = 287
        eps = 0.62197
        Cp = constants.isobaric_mass_heat_capacity
        VMR = self['H2O'][0, :]
        T = self['T'][0, :]
        lapse = g*(1 + Lv*VMR/R/T)/(Cp + Lv**2*VMR*eps/R/T**2)
        lapse_phlev = utils.calculate_halflevel_pressure(lapse.values)
        self['lapse'][0] = lapse_phlev

    def adjust(self, heatingrates, timestep, surface):

        self.moistlapse()

        # Apply heatingrates to temperature profile.
        self['T'] += heatingrates * timestep

        # Apply convective adjustment
        self.convective_adjustment(surface, timestep)

        # Preserve the initial relative humidity profile.
        # self.relative_humidity = self['initial_rel_humid'].values
        self.adjust_relative_humidity(heatingrates, timestep)

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()


class AtmosphereSlowMoistConvective(AtmosphereSlowConvective):

    def moistlapse(self):
        """Updates the atmospheric lapse rate for the convective adjustment
        according to the moist adiabat, which is calculated from the
        atmospheric temperature and humidity profiles. The lapse rate is in
        units of K/km.
        Parameters:
            a: atmosphere
        """
        g = constants.g
        Lv = 2501000
        R = 287
        eps = 0.62197
        Cp = constants.isobaric_mass_heat_capacity
        VMR = self['H2O'][0, :]
        T = self['T'][0, :]
        lapse = g*(1 + Lv*VMR/R/T)/(Cp + Lv**2*VMR*eps/R/T**2)
        lapse_phlev = utils.calculate_halflevel_pressure(lapse.values)
        self['lapse'][0] = lapse_phlev

    def adjust(self, heatingrates, timestep, surface):

        self.moistlapse()

        # Apply heatingrates to temperature profile.
        self['T'] += heatingrates * timestep

        # Apply convective adjustment
        self.convective_adjustment(surface, timestep)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()


class AtmosphereConUp(AtmosphereConvective):
    """
    Requires testing. Do not use.

    Atmosphere model with preserved RH and fixed temperature lapse rate,
    that includes a cooling term due to upwelling in the statosphere.
    """
    def upwelling_adjustment(self, ctop, timestep, w=0.0005):
        """Stratospheric cooling term parameterizing large-scale upwelling.

        Parameters:
            ctop (float): array index,
                the bottom level for the upwelling
                at and above this level, the upwelling is constant with height
            w (float): upwelling velocity
        """
        Cp = constants.isobaric_mass_heat_capacity
        g = constants.earth_standard_gravity

        actuallapse = self.get_lapse_rates()

        Q = -w * (-actuallapse + g / Cp)  # per second
        Q *= 24 * 60 * 60  # per day
        Q[:ctop] = 0

        self['T'] += Q * timestep

    def adjust(self, heatingrates, timestep, surface, w=0.0001, **kwargs):
        # TODO: Wrtie docstring.
        self['T'] += heatingrates * timestep

        if isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            # TODO: Output convective top for fixed_surface_temperature case
            ct = self.convective_adjustment_fixed_surface_temperature(surface=surface)
        else:
            ct, tdn, tdp = self.convective_top(surface=surface,
                                               timestep=timestep)

        self.upwelling_adjustment(ct, timestep, w)

        if not isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            self.convective_adjustment(surface=surface, timestep=timestep)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()


class AtmosphereConvectiveFlux(Atmosphere):
    """Convective flux."""
    def adjust(self, heatingrates, timestep, **kwargs):
        self['T'] += heatingrates * timestep

        Cp = conrad.constants.Cp
        p = self['plev'].values
        T = self['T'].values
        z = self['z'].values

        critical_lapse_rate = self.lapse[0, 1:-1]
        w = 0.01

        lapse_rate = -np.diff(T[0, :]) / np.diff(z[0, :])

        flux_divergence = w * (lapse_rate - critical_lapse_rate)
        dT = flux_divergence * timestep * 24 * 3600
        self['T'].values[0, :-1] += dT.values

        print(dT.values)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()
