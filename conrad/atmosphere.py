# -*- coding: utf-8 -*-
"""Module containing classes describing different atmosphere models.
"""

__all__ = [
    'Atmosphere',
    'AtmosphereFixedVMR',
    'AtmosphereFixedRH',
    'AtmosphereConvective',
    'AtmosphereMoistConvective',
    'AtmosphereConUp',
]


import abc
import collections
import logging

import typhon
import netCDF4
import numpy as np
from xarray import Dataset, DataArray

import conrad
from conrad import constants
from conrad import utils


logger = logging.getLogger()

atmosphere_variables = [
    'T',
    'H2O',
    'N2O',
    'O3',
    'CO2',
    'CO',
    'CH4',
]


class Atmosphere(Dataset, metaclass=abc.ABCMeta):
    """Abstract base class to define requirements for atmosphere models."""
    @abc.abstractmethod
    def adjust(self, heatingrate, timestep, **kwargs):
        """Adjust atmosphere according to given heatingrate."""

    @classmethod
    def from_atm_fields_compact(cls, atmfield, **kwargs):
        """Convert an ARTS atm_fields_compact [0] into an atmosphere.

        [0] http://arts.mi.uni-hamburg.de/docserver-trunk/variables/atm_fields_compact

        Parameters:
            atmfield (typhon.arts.types.GriddedField4): A compact set of
                atmospheric fields.
        """
        # Create a Dataset with time and pressure dimension.
        plev = atmfield.grids[1]
        phlev = utils.calculate_halflevel_pressure(plev)
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        },
                **kwargs,
                )

        for var in atmosphere_variables:
            # Get ARTS variable name from variable description.
            arts_key = constants.variable_description[var].get('arts_name')

            # Extract profile from atm_fields_compact
            profile = typhon.arts.atm_fields_compact_get(
                [arts_key], atmfield).squeeze()

            d[var] = DataArray(profile[np.newaxis, :], dims=('time', 'plev',))

        # Calculate the geopotential height.
        d.calculate_height()

        # Append variable descriptions to the Dataset.
        utils.append_description(d)

        return d

    @classmethod
    def from_dict(cls, dictionary, **kwargs):
        """Create an atmosphere model from dictionary values.

        Parameters:
            dictionary (dict): Dictionary containing ndarrays.
        """
        # TODO: Currently working for good-natured dictionaries.
        # Consider allowing a more flexibel user interface.

        # Create a Dataset with time and pressure dimension.
        plev = dictionary['plev']
        phlev = utils.calculate_halflevel_pressure(plev)
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        },
                **kwargs,
                )

        for var in atmosphere_variables:
            d[var] = DataArray(dictionary[var], dims=('time', 'plev',))

        # Calculate the geopotential height.
        d.calculate_height()

        # Append variable descriptions to the Dataset.
        utils.append_description(d)

        return d

    @classmethod
    def from_netcdf(cls, ncfile, timestep=-1, **kwargs):
        """Create an atmosphere model from a netCDF file.

        Parameters:
            ncfile (str): Path to netCDF file.
            timestep (int): Timestep to read (default is last timestep).
        """
        data = netCDF4.Dataset(ncfile).variables

        # Create a Dataset with time and pressure dimension.
        plev = data['plev'][:]
        phlev = utils.calculate_halflevel_pressure(plev)
        time = data['time'][[timestep]]
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        },
                **kwargs,
                )

        for var in atmosphere_variables:
            d[var] = DataArray(data[var][[timestep], :], dims=('time', 'plev',))

        # Calculate the geopotential height.
        d.calculate_height()

        # Append variable descriptions to the Dataset.
        utils.append_description(d)

        return d

    # TODO: This function could handle the nasty time dimension in the future.
    # Allowing to set two-dimensional variables using a 1d-array, if one
    # coordinate has the dimension one.
    def set(self, variable, value):
        """Set the values of a variable.

        Parameters:
            variable (str): Variable key.
            value (float or ndarray): Value to assign to the variable.
                If a float is given, all values are filled with it.
        """
        if isinstance(value, collections.Container):
            self[variable].values[0, :] = value
        else:
            self[variable].values.fill(value)

    def get_values(self, variable, keepdims=True):
        """Get values of a given variable.

        Parameters:
            variable (str): Variable key.
            keepdims (bool): If this is set to False, single-dimensions are
                removed. Otherwise dimensions are keppt (default).

        Returns:
            ndarray: Array containing the values assigned to the variable.
        """
        if keepdims:
            return self[variable].values
        else:
            return self[variable].values.ravel()

    def calculate_height(self):
        """Calculate the geopotential height."""
        g = constants.earth_standard_gravity

        plev = self['plev'].values  # Air pressure at full-levels.
        phlev = self['phlev'].values  # Air pressure at half-levels.
        T = self['T'].values  # Air temperature at full-levels.

        # Calculate the air density from current atmospheric state.
        rho = typhon.physics.density(plev, T)

        # Use the hydrostatic equation to calculate geopotential height from
        # given pressure, density and gravity.
        z = np.cumsum(-np.diff(phlev) / (rho * g))

        # If height is already in Dataset, update its values.
        if 'z' in self:
            self['z'].values[0, :] = np.cumsum(-np.diff(phlev) / (rho * g))
        # Otherwise create the DataArray.
        else:
            self['z'] = DataArray(z[np.newaxis, :], dims=('time', 'plev'))


    @property
    def relative_humidity(self):
        """Return the relative humidity of the current atmospheric state."""
        vmr, p, T = self['H2O'], self['plev'], self['T']
        return typhon.atmosphere.relative_humidity(vmr, p, T)

    @relative_humidity.setter
    def relative_humidity(self, RH):
        """Set the water vapor mixing ratio to match given relative humidity.

        Parameters:
            RH (ndarray or float): Relative humidity.
        """
        logger.debug('Adjust VMR to preserve relative humidity.')
        self['H2O'].values = typhon.atmosphere.vmr(RH, self['plev'], self['T'])

    def get_lapse_rates(self):
        """Calculate the temperature lapse rate at each level."""
        lapse_rate = np.diff(self['T'][0, :]) / np.diff(self['z'][0, :])
        lapse_rate = typhon.math.interpolate_halflevels(lapse_rate)
        lapse_rate = np.append(lapse_rate[0], lapse_rate)
        return np.append(lapse_rate, lapse_rate[-1])

    def find_first_unstable_layer(self, critical_lapse_rate=-0.0065,
                                  pmin=10e2):
        lapse_rate = self.get_lapse_rates()
        for n in range(len(lapse_rate) - 1, 1, -1):
            if lapse_rate[n] < critical_lapse_rate and self['plev'][n] > pmin:
                return n

    @property
    def cold_point_index(self, pmin=1e2):
        """Return the pressure index of the cold point tropopause."""
        return int(np.argmin(self['T'][:, self['plev'] > pmin]))

    def apply_H2O_limits(self, vmr_max=1.):
        """Adjust water vapor VMR values to follow physical limitations.

        Parameters:
            vmr_max (float): Maximum limit for water vapor VMR.
        """
        # Keep water vapor VMR values above the cold point tropopause constant.
        i = self.cold_point_index
        self['H2O'].values[0, i:] = self['H2O'][0, i]

        # NOTE: This has currently no effect, as the vmr_max is set to 1.
        # Limit the water vapor mixing ratios to a given threshold.
        too_high =  self['H2O'].values > vmr_max
        self['H2O'].values[too_high] = vmr_max


class AtmosphereFixedVMR(Atmosphere):
    """Atmosphere model with fixed volume mixing ratio."""
    def adjust(self, heatingrate, timestep, **kwargs):
        """Adjust the temperature.

        Adjust the atmospheric temperature profile by simply adding the given
        heatingrates.

        Parameters:
            heatingrates (float or ndarray): Heatingrate [K /day].
            timestep (float): Width of a timestep [day].
        """
        self['T'] += heatingrate * timestep


class AtmosphereFixedRH(Atmosphere):
    """Atmosphere model with fixed relative humidity.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio.

    Parameters:
        heatingrates (float or ndarray): Heatingrate [K /day].
        timestep (float): Width of a timestep [day].
    """
    def adjust(self, heatingrate, timestep, **kwargs):
        """Adjust the temperature and preserve relative humidity.

        Parameters:
            heatingrate (float or ndarray):
                Heatingrate (already scaled with timestep) [K].
        """
        # Apply heatingrates to temperature profile.
        self['T'] += heatingrate * timestep

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()


class AtmosphereConvective(Atmosphere):
    """Atmosphere model with preserved RH and fixed temperature lapse rate.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio. In addition, a simple
    convection parameterization is used.

    Implementation of Sally's convection scheme.
    """
    def __init__(self, *args, lapse=0.0065, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(lapse, float):
            lapse_array = lapse * np.ones((1, self['plev'].size))
            self['lapse'] = DataArray(lapse_array, dims=('time', 'plev'))
        elif isinstance(lapse, np.ndarray):
            self['lapse'] = DataArray(lapse, dims=('time', 'plev'))
        
        utils.append_description(self)  # Append variable descriptions.
    
    def convective_adjustment_fixed_surface_temperature(self, surface):
        """
        Apply a convective adjustment assuming the surface temperture is fixed.
        """
        p = self['plev']
        phlev = self['phlev']
        dp = np.diff(phlev)
        T_rad = self['T'][0, :].copy()
        T_con = T_rad.copy()
        density = typhon.physics.density(p, T_rad)
        g = constants.g
        lapse = self.lapse
        lp = -lapse[0, :] / (g*density)
        
        for level in range(0, len(p)):
            lapse_sum = np.sum(lp[0:level]*dp[0:level])
            T_con[level] = float(surface.temperature) - lapse_sum
        
        ct = np.min([np.where(T_con < T_rad)]) - 1
        
        T_con[np.where(T_con < T_rad)] = T_rad[np.where(T_con < T_rad)]
        self['T'].values = T_con.values[np.newaxis, :]
        
        self['convective_top'] = DataArray([ct], dims=('time',))
    
    def convective_top(self, surface, timestep):
        """Find the top of the convective layer, so that energy is conserved.
        """
        p = self['plev']
        phlev = self['phlev']
        dp = np.diff(phlev)
        T_rad = self['T'][0, :]
        density = typhon.physics.density(p, T_rad)
        Cp = constants.isobaric_mass_heat_capacity
        g = constants.g
        
        #TODO
        #tau = np.zeros(z.shape)
        #tau[round(len(z)/8):] = np.arange(0, len(tau[round(len(z)/8):])/20, 0.05)
        #Cp *= np.exp(-tau/timestep) # effective heat capacity
        
        lapse = self.lapse
        T_rad_s = surface.temperature
        
        if isinstance(surface, conrad.surface.SurfaceHeatCapacity):
            density_s = surface.rho
            Cp_s = surface.cp
            dz_s = surface.dz
        else:
            raise Exception('The convective adjustment is only available for '
                            'surfaces with heat capacity.')

        start_index = self.find_first_unstable_layer()
        if start_index is None:
            # If no unstable layer ist found, do nothing.
            return None, None, None

        termdiff_neg = 0
        lp = -lapse[0, :] / (g*density)

        # Precalculate some cumulative sums in order to reduce computations
        # performed in the following loop.
        term1_arr = np.cumsum(Cp / g * dp)
        term3_arr = np.cumsum(Cp / g * T_rad.values * dp)

        # Loop over all atmospheric layers above the first unstable layers.
        for a in range(start_index, len(p)):
            term1 = T_rad[a] * term1_arr[a]
            term3 = term3_arr[a]
            # TODO: Here is quite possibly more potential for precalculation...
            term_s = dz_s*density_s*Cp_s*(T_rad[a]+np.sum(lp[0:a]*dp[0:a])-T_rad_s)
            inintegral = utils.revcumsum(lp.values[:a] * dp[:a])
            term2 = np.sum((Cp/g)*inintegral * dp[:a])
            if (-term1 - term2 + term3 + term_s) > 0:
                break
            termdiff_neg = -term1 - term2 + term3 + term_s
        termdiff_pos = -term1 - term2 + term3 + term_s
        
        self['convective_top'] = DataArray([a - 1], dims=('time',))
        
        return a-1, float(termdiff_neg), float(termdiff_pos)
    
    def energy_difference(self, T_con, T_rad, sst_con, sst_rad, dp, eff_Cp_s):
        """
        Lukas, this should not need to be a function of self. Please change
            this if you can :)
        
        Calculate the energy difference from zero of the convective adjustment.
        
        Parameters:
            T_con: adjusted temperature profile
            T_rad: radiative temperature profile 
                (from which the convective adjustment is made)
            sst_con: adjusted surface temperature
            sst_rad: radiative surface temperature
            dp: pressure thicknesses of levels
            eff_Cp_s: effective heat capacity of surface
        """
        Cp = constants.isobaric_mass_heat_capacity
        g = constants.g
    
        dT_con = T_con - T_rad
        dT_con_s = sst_con - sst_rad
        termdiff = - np.sum(Cp/g * dT_con * dp) + eff_Cp_s * dT_con_s
        
        return termdiff
    
    def readjust(self, ct, T_con, surface, surfacetemp, termdiff_neg, Tct_min, 
                 termdiff_pos, Tct_max):
        """
        Make an adjusment to the convective top temperature (and thus the whole
        convective profile), such that it is closer to being energy conserving.
        
        Parameters:
            ct: array index,
                the top level for the convective adjustment
            T_con: convectively adjusted temperature profile
            surface: surface with temperature from the radiative profile
            surfacetemp: convectively adjusted surface temperature
            termdiff_neg: difference from zero of the convective adjustment 
                integration when using a convective top temperature of 
                T[ct] = Tct_min
            Tct_min: lower bound for the temperature of the convective top
            termdiff_pos: difference from zero of the convective adjustment 
                integration when using a convective top temperature of
                T[ct] = Tct_max
            Tct_max: upper bound for the temperature of the convective top
        """
        T_rad = self['T'][0, :]
        T_con_new = T_con.copy()
        g = constants.g
        p = self.plev
        phlev = self.phlev
        dp = np.diff(phlev)
        density = typhon.physics.density(p, T_rad)
        lp = - self.lapse[0, :] / (g*density)
        
        eff_Cp_s = surface.rho * surface.cp * surface.dz
        
        termdiff = self.energy_difference(T_con, T_rad, surfacetemp,
                                          surface.temperature[0], dp, eff_Cp_s)
        
        if termdiff > 0:
            Tct_max = T_con[ct]
            termdiff_pos = termdiff
        else:
            Tct_min = T_con[ct]
            termdiff_neg = termdiff
        
        frct = -termdiff_neg / (termdiff_pos - termdiff_neg)
        
        # adjust convective top temperature
        if termdiff > 0:
            T_con_new[ct] -= (1 - frct) * (T_con[ct] - Tct_min)
        else:
            T_con_new[ct] += (Tct_max - T_con[ct])*frct
            
        # adjust surface temperature
        surfacetemp_new = T_con_new[ct] + np.sum(lp[0:ct]*dp[0:ct])
        # adjust temperature of other atmospheric layers
        T_con_new.values[0:ct] = (
            T_con_new.values[ct] + utils.revcumsum(lp.values[0:ct] * dp[0:ct])
        )

        return T_con_new, surfacetemp_new, termdiff_neg, Tct_min, termdiff_pos, Tct_max

    def convective_adjustment(self, ct, tdn, tdp, surface, timestep):
        """Apply the convective adjustment.

        Parameters:
            ct (float): array index,
                the top level for the convective adjustment
            tdn, tdp: differences from zero of the convective adjustment 
                integration when using a convective top at ct and ct+1 
                respectively and with the convective top temperature equal to
                the radiative temperature at that level
        """
        p = self['plev']
        phlev = self['phlev']
        dp = np.diff(phlev)
        T_rad = self['T'][0, :]
        T_con = T_rad.copy()
        lapse = self.lapse
        density = typhon.physics.density(p, T_rad)
        g = constants.g
        eff_Cp_s = surface.rho * surface.cp * surface.dz
        
        #TODO
        #tau = np.zeros(z.shape)
        #tau[round(len(z)/8):] = np.arange(0, len(tau[round(len(z)/8):])/20, 0.05)

        lp = -lapse[0, :] / (g*density)
        
        frct = - tdn / (tdp - tdn)
        levelup_T_at_ct = self['T'][0, ct+1] + lp[ct]*dp[ct]
        T_con[ct] += (levelup_T_at_ct - self['T'][0, ct])*frct
        # adjust surface temperature
        surfacetemp = T_con[ct] + np.sum(lp[0:ct]*dp[0:ct])
        # adjust temperature of other atmospheric layers
        T_con.values[0:ct] = (
            T_con.values[ct] + utils.revcumsum(lp.values[0:ct] * dp[0:ct])
        )
        
        Tct_max = levelup_T_at_ct
        Tct_min = T_rad[ct]
        sst_rad = surface['temperature'][0]
        energydiff = self.energy_difference(T_con, T_rad, surfacetemp, sst_rad,
                                            dp, eff_Cp_s)
        
        # adjust the convective profile: perform a maximum of 3 iterations or 
        # until the energy difference is sufficiently small
        counter = 0
        while counter < 3 and np.absolute(float(energydiff)) > 0.0001:
            newvals = self.readjust(ct, T_con, surface, surfacetemp, tdn, 
                                    Tct_min, tdp, Tct_max)
            T_con, surfacetemp, tdn, Tct_min, tdp, Tct_max = newvals
            energydiff = self.energy_difference(T_con, T_rad, surfacetemp,
                                                sst_rad, dp, eff_Cp_s)
            counter += 1
        
        # save new temperature profile
        surface['temperature'][0] = surfacetemp
        self['T'].values = T_con.values[np.newaxis, :]
        

    def adjust(self, heatingrates, timestep, surface):
        """Adjust the temperature and preserve relative humidity and lapse rate.

        Parameters:
            heatingrates (float or ndarray): Heatingrate [K /day].
            timestep (float): Width of a timestep [day].
        """
        # Apply heatingrates to temperature profile.
        self['T'] += heatingrates * timestep
        
        if isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            self.convective_adjustment_fixed_surface_temperature(
                surface=surface,
            )
        else:
            # Search for the top of convection.
            ct, tdn, tdp = self.convective_top(surface=surface, timestep=timestep)

            # If a convective top is found, apply the convective adjustment.
            if ct is not None:
                self.convective_adjustment(
                    ct,
                    tdn, tdp,
                    surface=surface,
                    timestep=timestep,
                )

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()

        
class AtmosphereConUp(AtmosphereConvective):
    """Atmosphere model with preserved RH and fixed temperature lapse rate,
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

        Q = -w * (-actuallapse + g / Cp) # per second
        Q *= 24*60*60 # per day
        Q[:ctop] = 0

        self['T'] += Q * timestep

    def adjust(self, heatingrates, timestep, surface, w=0.0001, **kwargs):
        #TODO: Wrtie docstring.
        self['T'] += heatingrates * timestep

        if isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            ct = self.convective_adjustment_fixed_surface_temperature(surface=surface)
        else:
            ct, frct = self.convective_top(surface=surface, timestep=timestep)

        self.upwelling_adjustment(ct, timestep, w)
        
        if not isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            self.convective_adjustment(ct, frct, surface=surface, timestep=timestep)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()


class AtmosphereMoistConvective(AtmosphereConUp):
    """Atmosphere model with preserved RH and a temperature and humidity
    -dependent lapse rate.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio. In addition, a convection
    parameterization is used, which sets the lapse rate to the moist adiabat,
    calculated from the previous temperature and humidity profiles.
    """     
    
    def moistlapse(self):
        """ 
        Calculates the moist lapse rate from the atmospheric temperature and 
            humidity profiles 
        Parameters:
            a: atmosphere
        """
        g = constants.g
        Lv = 2501000
        R = 287
        eps = 0.62197
        Cp = conrad.constants.isobaric_mass_heat_capacity
        VMR = self['H2O'][0, :]
        T = self['T'][0, :]
        lapse = g*(1 + Lv*VMR/R/T)/(Cp + Lv**2*VMR*eps/R/T**2)
        self.lapse = lapse
    
    def adjust(self, heatingrates, timestep, surface, w=0, **kwargs):
        # TODO: Write docstring.

        # TODO: Calculate moist lapse rates instead of hardcoding them.
        # calculate lapse rate based on previous con-rad state.

        self.moistlapse # set lapse rate
        
        self['T'] += heatingrates * timestep
        
        if isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            ct = self.convective_adjustment_fixed_surface_temperature(surface=surface)
        else:
            ct, frct = self.convective_top(surface=surface, timestep=timestep)

        if w != 0:
            self.upwelling_adjustment(ct, timestep, w)
        
        if not isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            self.convective_adjustment(ct, frct, surface=surface, timestep=timestep)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()
