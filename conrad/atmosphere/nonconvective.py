# -*- coding: utf-8 -*-
import logging

from conrad.atmosphere.abc import Atmosphere

logger = logging.getLogger()


__all__ = [
    'AtmosphereFixedVMR',
    'AtmosphereFixedRH',
]


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
        # Apply heatingrates to temperature profile.
        self['T'] += heatingrate * timestep

        # Calculate the geopotential height field.
        self.calculate_height()


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
