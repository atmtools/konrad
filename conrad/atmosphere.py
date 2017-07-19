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
    'z',
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
        phlev = utils.calculate_halflevels(plev)
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

        utils.append_description(d)  # Append variable descriptions.

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
        phlev = utils.calculate_halflevels(plev)
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        },
                **kwargs,
                )

        for var in atmosphere_variables:
            d[var] = DataArray(dictionary[var], dims=('time', 'plev',))

        utils.append_description(d)  # Append variable descriptions.

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
        phlev = utils.calculate_halflevels(plev)
        time = data['time'][[timestep]]
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        },
                **kwargs,
                )

        for var in atmosphere_variables:
            d[var] = DataArray(data[var][[timestep], :], dims=('time', 'plev',))

        utils.append_description(d)  # Append variable descriptions.

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

    def apply_H2O_limits(self):
        """Adjust water vapor VMR values to follow physical limitations."""
        # Keep water vapor VMR values above the cold point tropopause constant.
        i = self.cold_point_index
        self['H2O'].values[0, i:] = self['H2O'][0, i]

        # TODO: Set an upper VMR value at some point.
        # # Do not allow mixing ratios above 8 percent.
        # too_high =  self['H2O'].values > 0.05
        # self['H2O'].values[too_high] = 0.05


class AtmosphereFixedVMR(Atmosphere):
    """Atmosphere model with fixed volume mixing ratio."""
    def adjust(self, heatingrate, timestep, **kwargs):
        """Adjust the temperature.

        Adjust the atmospheric temperature profile by simply adding the given
        heatingrates.

        Parameters:
            heatingrate (float or ndarray):
                Heatingrate (already scaled with timestep) [K].
        """
        self['T'] += heatingrate * timestep


class AtmosphereFixedRH(Atmosphere):
    """Atmosphere model with fixed relative humidity.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio.
    """
    def adjust(self, heatingrate, timestep, **kwargs):
        """Adjust the temperature and preserve relative humidity.

        Parameters:
            heatingrate (float or ndarray):
                Heatingrate (already scaled with timestep) [K].
        """
        self['T'] += heatingrate * timestep  # adjust temperature profile.

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        self.apply_H2O_limits()  # adjust stratospheric VMR values.


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
            self['lapse'] = lapse
        elif isinstance(lapse, np.ndarray):
            self['lapse'] = DataArray(lapse, dims=('time', 'plev'))
        
        utils.append_description(self)  # Append variable descriptions.
    
    def convective_adjustment_fixed_surface_temperature(self, surface):
        z_s = surface.height
        z = self['z'][0, :]
        T_rad = self['T'][0, :].copy()
        lapse = self.lapse
        
        if lapse.shape == (): # Fixed lapse rate case
            T_con = float(surface.temperature) - (z - z_s)*lapse
        
        else: # Lapse rate varies with height
            for level in range(0, len(z)):
                lapse_sum = np.trapz(lapse[0:level], z[0:level])
                T_con[level] = float(surface.temperature) + lapse_sum
        
        T_con[np.where(T_con < T_rad)] = T_rad[np.where(T_con < T_rad)]
        self['T'].values = T_con.values[np.newaxis, :]
        
        #return np.min(np.where(T_con < T_rad)) - 1
    
    def convective_top(self, surface, timestep):
        """Find the top of the convective layer, so that energy is conserved.

        Parameters:
            lapse (float or array): lapse rate value to adjust to
            timestep (float): for the case of a surface with no heat capacity,
                to calculate the energy emitted in that time
        """
        p = self['plev']
        z = self['z'][0, :]
        T_rad = self['T'][0, :]
        density = typhon.physics.density(p, T_rad)
        Cp = constants.isobaric_mass_heat_capacity
        
        #TODO
        #tau = np.zeros(z.shape)
        #tau[round(len(z)/8):] = np.arange(0, len(tau[round(len(z)/8):])/20, 0.05)
        #Cp *= np.exp(-tau/timestep) # effective heat capacity
        
        lapse = self.lapse
        z_s = surface.height
        T_rad_s = surface.temperature
        
        if isinstance(surface, conrad.surface.SurfaceHeatCapacity):
            density_s = surface.rho
            Cp_s = surface.cp
            dz_s = surface.dz    
        
        start_index = self.find_first_unstable_layer()
        if start_index is None:
            return None, None

        # Fixed lapse rate case
        if np.size(lapse) == 1:
            lp = float(lapse)
            
            termdiff = 0
            for a in range(start_index, len(z)):
                term1 = (T_rad[a]+z[a]*lp)*np.trapz((density*Cp)[:a+1], z[:a+1])
                term2 = np.trapz((density*Cp*(T_rad+z*lp))[:a+1], z[:a+1])
                if isinstance(surface, conrad.surface.SurfaceHeatCapacity):
                    term_s = dz_s*density_s*Cp_s*(T_rad[a]-(z_s-z[a])*lp-T_rad_s)
                elif isinstance(surface, conrad.surface.SurfaceNoHeatCapacity):
                    sigma = constants.stefan_boltzmann
                    T_adj_s = T_rad[a] - (z_s-z[a])*lp
                    term_s = sigma*(T_adj_s**4 - T_rad_s**4)*timestep
                if (term1 - term2 + term_s) > 0:
                    break
                termdiff = term1 - term2 + term_s
            frct = -termdiff / ((term1-term2+term_s)-termdiff)
            return a-1, float(frct)
        
        # Lapse rate varies with height
        # TODO: add case for surface with no heat capacity
        else:
            termdiff = 0
            lp = lapse[0, :]
            for a in range(start_index, len(z)):
                term1 = T_rad[a]*np.trapz((density*Cp)[:a+1], z[:a+1])
                term3 = np.trapz((density*Cp*T_rad)[:a+1], z[:a+1])
                term_s = dz_s*density_s*Cp_s*(T_rad[a]+np.trapz(lp[0:a+1], z[0:a+1])-T_rad_s)
                #term_s = 0
                inintegral = np.zeros((a+1,))
                for b in range(0, a+1):
                    inintegral[b] = np.trapz(lp[b:a+1], z[b:a+1])
                term2 = np.trapz((density*Cp)[:a+1]*inintegral, z[:a+1])
                if (term1 + term2 - term3 + term_s) > 0:
                    break
                termdiff = term1 + term2 - term3 + term_s
            frct = -termdiff / ((term1+term2-term3+term_s)-termdiff)
            return a-1, float(frct)

    def convective_adjustment(self, ct, frct, surface, timestep):
        """Apply the convective adjustment.

        Parameters:
            ct (float): array index,
                the top level for the convective adjustment
            frct (float):
                fraction: energy imbalance over energy difference between 
                two convective profiles
            lapse (float or array):
                adjust to this lapse rate value
        """
        z = self['z'][0, :]
        T_rad = self['T'][0, :]
        T_con = T_rad.copy()
        lapse = self.lapse
        z_s = surface.height
        
        #tau = np.zeros(z.shape)
        #tau[round(len(z)/8):] = np.arange(0, len(tau[round(len(z)/8):])/20, 0.05)
        # Fixed lapse rate case
        if np.size(lapse) == 1:
            lp = float(lapse)
            # adjust temperature at convective top
            levelup_T_at_ct = self['T'][0, ct+1] - lp*(z[ct] - z[ct+1])
            T_con[ct] += (levelup_T_at_ct - self['T'][0, ct])*frct
            # adjust surface temperature
            surface['temperature'][0] = T_con[ct] - (z_s-z[ct])*lp
            # adjust temperature of other atmospheric layers
            T_con[:ct] = T_con[ct] - (z[:ct]-z[ct])*lp
            
            #TODO
            #expo = np.exp(-tau/timestep)
            #T_con[:ct] = T_con[:ct]*expo[:ct] + T_rad[:ct]*(1-expo[:ct])
        
        # Lapse rate varies with height
        else:
            lp = lapse[0, :]
            # adjust temperature at convective top
            # TODO: What index do I need for lapse here?
            levelup_T_at_ct = self['T'][0, ct+1] - lp[ct+1]*(z[ct]-z[ct+1])
            T_con[ct] += (levelup_T_at_ct - self['T'][0, ct])*frct
            # adjust surface temperature
            surface['temperature'][0] = T_con[ct] + np.trapz(lp[0:ct+1], z[0:ct+1])
            # adjust temperature of other atmospheric layers
            for level in range(0, ct):
                lapse_sum = np.trapz(lp[level:ct+1], z[level:ct+1])
                T_con[level] = T_con[ct] + lapse_sum

        self['T'].values = T_con.values[np.newaxis, :]

    def adjust(self, heatingrates, timestep, surface):
        self['T'] += heatingrates * timestep
        
        if isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            self.convective_adjustment_fixed_surface_temperature(surface=surface)
        else:
            ct, frct = self.convective_top(surface=surface, timestep=timestep)
            self.convective_adjustment(ct, frct, surface=surface, timestep=timestep)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        self.apply_H2O_limits()  # adjust stratospheric VMR values.

        
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
        self['T'] += heatingrates * timestep

        if isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            ct = self.convective_adjustment_fixed_surface_temperature(surface=surface)
        else:
            ct, frct = self.convective_top(surface=surface, timestep=timestep)

        self.upwelling_adjustment(ct, timestep, w)
        
        if not isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            self.convective_adjustment(ct, frct, surface=surface)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        self.apply_H2O_limits()  # adjust stratospheric VMR values.


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
            self.convective_adjustment(ct, frct, surface=surface)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        self.apply_H2O_limits()  # adjust stratospheric VMR values.
