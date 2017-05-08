# -*- coding: utf-8 -*-
"""Module containing classes describing different atmosphere models.
"""

__all__ = [
    'Atmosphere',
    'AtmosphereFixedVMR',
    'AtmosphereFixedRH',
    'AtmosphereConvective',
]


import abc
import collections
import logging

import typhon
import numpy as np
from xarray import Dataset, DataArray


logger = logging.getLogger()

# Dictionary containing information on atmospheric constituents.
# Variables listed in this dictionary are **required** parts of every
# atmosphere model!
variable_description = {
    'T': {'units': 'K',
          'standard_name': 'air_temperature',
          'arts_name': 'T',
          },
    'z': {'units': 'm',
          'standard_name': 'height',
          'arts_name': 'z',
          },
    'H2O': {'units': '1',
            'standard_name': 'humidity_mixing_ratio',
            'arts_name': 'abs_species-H2O',
            },
    'N2O': {'units': '1',
            'standard_name': 'nitrogene_mixing_ratio',
            'arts_name': 'abs_species-N2O',
            },
    'O3': {'units': '1',
           'standard_name': 'ozone_mixing_ratio',
           'arts_name': 'abs_species-O3',
           },
    'CO2': {'units': '1',
            'standard_name': 'carbon_dioxide_mixing_ratio',
            'arts_name': 'abs_species-CO2',
            },
    'CO': {'units': '1',
           'standard_name': 'carbon_monoxide_mixing_ratio',
           'arts_name': 'abs_species-CO',
           },
    'CH4': {'units': '1',
            'standard_name': 'methane_mixing_ratio',
            'arts_name': 'abs_species-CH4',
            },
}


class Atmosphere(Dataset, metaclass=abc.ABCMeta):
    """Abstract base class to define requirements for atmosphere models."""
    @abc.abstractmethod
    def adjust(self, heatingrate):
        """Adjust atmosphere according to given heatingrate."""

    @classmethod
    def from_atm_fields_compact(cls, atmfield):
        """Convert an ARTS atm_fields_compact [0] into an atmosphere.

        [0] http://arts.mi.uni-hamburg.de/docserver-trunk/variables/atm_fields_compact

        Parameters:
            atmfield (typhon.arts.types.GriddedField4): A compact set of
                atmospheric fields.
        """
        # Create a Dataset with time and pressure dimension.
        d = cls(coords={'plev': atmfield.grids[1], 'time': [0]})

        # TODO: Maybe introduce another variables `required_vars` to not loop
        # over all variable descriptions. This would allow unused descriptions.
        for var, desc in variable_description.items():
            atmfield_data = typhon.arts.atm_fields_compact_get(
                [desc['arts_name']], atmfield).squeeze()
            darray = DataArray(atmfield_data[np.newaxis, :],
                               dims=('time', 'plev',))
            darray.attrs['standard_name'] = desc['standard_name']
            darray.attrs['units'] = desc['units']
            d[var] = darray

        d['plev'].attrs['standard_name'] = 'air_pressure'
        d['plev'].attrs['units'] = 'Pa'
        d['time'].attrs['standard_name'] = 'time'
        d['time'].attrs['units'] = 'hours since 0001-01-01 00:00:00.0'
        d['time'].attrs['calender'] = 'gregorian'

        return d

    @classmethod
    def from_dict(cls, dict):
        """Create an atmosphere model from dictionary values.

        Parameters:
            d (dict): Dictionary containing ndarrays.
        """
        # TODO: Currently working for good-natured dictionaries.
        # Consider allowing a more flexibel user interface.

        # Create a Dataset with time and pressure dimension.
        d = cls(coords={'plev': dict['plev'], 'time': [0]})

        for var, desc in variable_description.items():
            darray = DataArray(dict[var], dims=('time', 'plev',))
            darray.attrs['standard_name'] = desc['standard_name']
            darray.attrs['units'] = desc['units']
            d[var] = darray

        d['plev'].attrs['standard_name'] = 'air_pressure'
        d['plev'].attrs['units'] = 'Pa'
        d['time'].attrs['standard_name'] = 'time'
        d['time'].attrs['units'] = 'hours since 0001-01-01 00:00:00.0'
        d['time'].attrs['calender'] = 'gregorian'

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
            self[variable].values = value
        else:
            self[variable].values.fill(value)

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
        self['H2O'] = typhon.atmosphere.vmr(RH, self['plev'], self['T'])
        self['H2O'][0, self['H2O'][0, :] < 3e-6] = 3e-6
        self['H2O'][0, self['plev'] < 50e2] = 3e-6

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

    # def convective_adjustment(self, critical_lapse_rate=-0.0065, pmin=10e2):
    #     i = self.find_first_unstable_layer(
    #             critical_lapse_rate=critical_lapse_rate,
    #             pmin=pmin)

    #     if i is not None:
    #         self['T'][0, :i] = (self['T'][0, i] + critical_lapse_rate
    #                             * (self['z'][0, :i] - self['z'][0, i]))


class AtmosphereFixedVMR(Atmosphere):
    """Atmosphere model with fixed volume mixing ratio."""
    def adjust(self, heatingrate):
        """Adjust the temperature.

        Adjust the atmospheric temperature profile by simply adding the given
        heatingrates.

        Parameters:
            heatingrate (float or ndarray):
                Heatingrate (already scaled with timestep) [K].
        """
        self['T'] += heatingrate


class AtmosphereFixedRH(Atmosphere):
    """Atmosphere model with fixed relative humidity.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio.
    """
    def adjust(self, heatingrate):
        """Adjust the temperature and preserve relative humidity.

        Parameters:
            heatingrate (float or ndarray):
                Heatingrate (already scaled with timestep) [K].
        """
        # Store initial relative humidty profile.
        RH = self.relative_humidity

        self['T'] += heatingrate  # adjust temperature profile.

        self.relative_humidity = RH  # reset original RH profile.


class AtmosphereConvective(Atmosphere):
    """Atmosphere model with preserved RH and fixed temperature lapse rate.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio. In addition, a simple
    convection parameterization is used.

    Implementation of Sally's convection scheme.
    """
    def convective_adjustment(self, lapse=0.0065):
        p = self['plev']
        z = self['z'][0, :]
        T_rad = self['T'][0, :]
        density = typhon.physics.density(p, T_rad)
        Cp = 1003.5

        # Fixed lapse rate case
        start_index = 1
        start_index = self.find_first_unstable_layer()
        if start_index is None:
            return

        for a in range(start_index, len(z)):
            term2 = np.trapz((density*Cp*(T_rad+z*lapse))[:a], z[:a])
            term1 = (T_rad[a]+z[a]*lapse)*np.trapz((density*Cp)[:a], z[:a])
            if (term1 - term2) > 0:
                break

        T_con = T_rad.copy()
        for level in range(0, a):
            T_con[level] = T_rad[a] - (z[level]-z[a])*lapse

        self['T'].values = T_con.values[np.newaxis, :]

    def adjust(self, heatingrates):
        RH = self.relative_humidity

        self['T'] += heatingrates

        self.convective_adjustment()

        self.relative_humidity = RH  # reset original RH profile.
