# -*- coding: utf-8 -*-
"""Module containing classes describing different atmosphere models.
"""

__all__ = [
    'Atmosphere',
    'AtmosphereFixedVMR',
    'AtmosphereFixedRH',
]


import abc
import logging

import typhon
import numpy as np
from xarray import Dataset, DataArray


logger = logging.getLogger()

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
    """Abstract base class to define common requirements for atmosphere models.
    """
    @abc.abstractmethod
    def adjust(self, heatingrate):
        """Adjust atmosphere according to given heatingrate."""

    @classmethod
    def from_atm_fields_compact(cls, atmfield):
        """Convert an atm_fields_compact into an atmosphere.

        Parameters:
            atmfield (typhon.arts.types.GriddedField4): Atmosphere field.
        """
        # Create a Dataset with time and pressure dimension.
        d = cls(coords={'plev': atmfield.grids[1], 'time': [0]})

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


class AtmosphereFixedVMR(Atmosphere):
    """Describes an atmosphere with fixed volume mixing ratio."""
    def adjust(self, heatingrate):
        """Adjust the temperature."""
        self['T'] += heatingrate


class AtmosphereFixedRH(Atmosphere):
    """Describes an atmosphere with fixed relative humidity."""
    def adjust_vmr(self, RH):
        logger.debug('Adjust VMR to preserve relative humidity.')
        self['H2O'] = typhon.atmosphere.vmr(RH, self['plev'], self['T'])

    def adjust(self, heatingrate):
        """Adjust the temperature while and preserve relative humidity."""
        # Store initial relative humidty profile.
        RH = typhon.atmosphere.relative_humidity(self['H2O'],
                                                 self['plev'],
                                                 self['T'])

        self['T'] += heatingrate  # adjust temperature profile.

        self.adjust_vmr(RH)  # adjust VMR to preserve original RH profile.
