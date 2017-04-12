# -*- coding: utf-8 -*-
"""Module containing classes describing different atmosphere models.
"""

__all__ = [
    'Atmosphere',
    'AtmosphereFixedVMR',
    'AtmosphereFixedRH',
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

    def adjust_vmr(self, RH):
        """Set the water vapor mixing ratio to match given relative humidity.

        Parameters:
            RH (ndarray or float): Relative humidity.
        """
        logger.debug('Adjust VMR to preserve relative humidity.')
        self['H2O'] = typhon.atmosphere.vmr(RH, self['plev'], self['T'])

    def convective_adjustment(self):
        """Adjusted the vertical temperature profile to the dry adiatie."""
        pass


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
        RH = typhon.atmosphere.relative_humidity(self['H2O'],
                                                 self['plev'],
                                                 self['T'])

        self['T'] += heatingrate  # adjust temperature profile.

        self.adjust_vmr(RH)  # adjust VMR to preserve original RH profile.


class AtmosphereConvective(Atmosphere):
    """Atmosphere model with preserved RH and fixed temperature lapse rate.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio. In addition, a simple
    convection parameterization is used.
    """
    def adjust(self, heatingrate):
        """Adjust the temperature and preserve relative humidity.

        Parameters:
            heatingrate (float or ndarray):
                Heatingrate (already scaled with timestep) [K].
        """
        # Store initial relative humidty profile.
        RH = typhon.atmosphere.relative_humidity(self['H2O'],
                                                 self['plev'],
                                                 self['T'])

        self['T'] += heatingrate  # adjust temperature profile.

        self.convective_adjustment()  # adjust temperature lapse rate.

        self.adjust_vmr(RH)  # adjust VMR to preserve original RH profile.
