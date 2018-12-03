"""Define the abstract Radiation base class. """
import abc
import logging

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from konrad import constants
from konrad.component import Component
from konrad.utils import append_description


logger = logging.getLogger(__name__)

__all__ = [
    'Radiation',
]

# Subclasses of `Radiation` need to define a `Radiation.calc_radiation()`
# method that returns a `xr.Dataset` containing at least following variables:
REQUIRED_VARIABLES = [
    'net_htngrt',
    'lw_flxd',
    'lw_flxu',
    'sw_flxd',
    'sw_flxu',
]


class Radiation(Component, metaclass=abc.ABCMeta):
    """Abstract base class to define requirements for radiation models."""
    def __init__(self, zenith_angle=47.88, diurnal_cycle=False, bias=None):
        """Return a radiation model.

        Parameters:
            zenith_angle (float): Zenith angle of the sun.
                The default angle of 47.88 degree results in 342 W/m^2
                solar insolation at the top of the atmosphere when used
                together with a solar constant of 510 W/m^2.
            diurnal_cycle (bool): Toggle diurnal cycle of solar angle.
            bias (dict-like): A dict-like object that stores bias
                corrections for the diagnostic variable specified by its key,
                e.g. `bias = {'net_htngrt': 2}`.
        """
        super().__init__()

        self.zenith_angle = zenith_angle
        self.diurnal_cycle = diurnal_cycle

        if diurnal_cycle:
            self.current_solar_angle = 0
        else:
            self.current_solar_angle = self.zenith_angle

        self._bias = bias

        self['lw_htngrt'] = (('time', 'plev'), None)
        self['lw_htngrt_clr'] = (('time', 'plev'), None)
        self['lw_flxu'] = (('time', 'phlev'), None)
        self['lw_flxd'] = (('time', 'phlev'), None)
        self['lw_flxu_clr'] = (('time', 'phlev'), None)
        self['lw_flxd_clr'] = (('time', 'phlev'), None)
        self['sw_htngrt'] = (('time', 'plev'), None)
        self['sw_htngrt_clr'] = (('time', 'plev'), None)
        self['sw_flxu'] = (('time', 'phlev'), None)
        self['sw_flxd'] = (('time', 'phlev'), None)
        self['sw_flxu_clr'] = (('time', 'phlev'), None)
        self['sw_flxd_clr'] = (('time', 'phlev'), None)

        self['net_htngrt'] = (('time', 'plev'), None)
        self['toa'] = (('time',), None)

    @abc.abstractmethod
    def calc_radiation(self, atmosphere, surface, cloud):
        pass

    def update_heatingrates(self, atmosphere, surface, cloud):
        """Returns `xr.Dataset` containing radiative transfer results."""
        self.calc_radiation(atmosphere, surface, cloud)

        # self.correct_bias(rad_dataset)

        self.derive_diagnostics()

    @staticmethod
    def check_dataset(dataset):
        """Check if a given dataset contains all required variables."""
        for key in REQUIRED_VARIABLES:
            if key not in dataset.variables:
                raise KeyError(
                    f'"{key}" not present in radiative transfer results.'
                )

    def correct_bias(self, dataset):
        """Apply bias correction."""
        # Interpolate biases passed as `xr.Dataset`.
        if isinstance(self._bias, xr.Dataset):
            bias_dict = {}
            for key in self._bias.data_vars:
                zdim = self._bias[key].dims[0]
                x = self._bias[zdim].values
                y = self._bias[key].values

                f_interp = interp1d(x, y, fill_value='extrapolate')
                bias_dict[key] = f_interp(dataset[zdim].values)

            self._bias = bias_dict

        if self._bias is not None:
            for key, value in self._bias.items():
                if key not in dataset.indexes:
                    dataset[key] -= value


    def derive_diagnostics(self):
        """Derive diagnostic variables from radiative transfer results."""
        # Net heating rate.
        self['net_htngrt'] = self['lw_htngrt'] + self['sw_htngrt']

        # Radiation budget at top of the atmosphere (TOA).
        self['toa'] = (
            (self['sw_flxd'][:, -1] + self['lw_flxd'][:, -1]) -
            (self['sw_flxu'][:, -1] + self['lw_flxu'][:, -1])
        )

    @staticmethod
    def heatingrates_from_fluxes(pressure, downward_flux, upward_flux):
        """Calculate heating rates from radiative fluxes.

        Parameters:
            pressure (ndarray): Pressure half-levels [Pa].
            downward_flux (ndarray): Downward radiative flux [W/m^2].
            upward_flux (ndarray): Upward radiative flux [W/m^2].

        Returns:
            ndarray: Radiative heating rate [K/day].
        """
        c_p = constants.isobaric_mass_heat_capacity
        g = constants.earth_standard_gravity

        q = g / c_p * np.diff(upward_flux - downward_flux) / np.diff(pressure)
        q *= 3600 * 24

        return q

    def adjust_solar_angle(self, time):
        """Adjust the zenith angle of the sun according to time of day.

        Parameters:
            time (float): Current time [days].
        """
        # When the diurnal cycle is disabled, use the constant zenith angle.
        if not self.diurnal_cycle:
            self.current_solar_angle = self.zenith_angle
            return

        # The local zenith angle, calculated from the latitude and longitude.
        # Seasons are not considered.
        solar_angle = np.rad2deg(np.arccos(
            np.cos(np.deg2rad(self.zenith_angle)) * 
            np.cos(2 * np.pi * time)))

        self.current_solar_angle = solar_angle
