# -*- coding: utf-8 -*-
"""Module containing classes describing different radiation models.
"""
import abc
import logging

from xarray import Dataset
import numpy as np

from . import utils
from conrad.utils import append_description


logger = logging.getLogger()

__all__ = [
    'Radiation',
    'PSRAD',
]


class Radiation(metaclass=abc.ABCMeta):
    """Abstract base class to define requirements for radiation models."""
    def __init__(self, zenith_angle=47.88, diurnal_cycle=False):
        """Return a radiation model.

        Parameters:
            zenith_angle (float): Zenith angle of the sun.
                The default angle of 47.88 degree results in 342 W/m^2
                solar insolation at the top of the atmosphere when used
                together with a solar constant of 510 W/m^2.
            diurnal_cycle (bool): Toggle diurnal cycle of solar angle.
        """
        self.zenith_angle = zenith_angle
        self.diurnal_cycle = diurnal_cycle

        self.current_solar_angle = 0

    @abc.abstractmethod
    def get_heatingrates(self, atmosphere):
        """Returns the shortwave, longwave and net heatingrates."""
        pass

    def adjust_solar_angle(self, time):
        """Adjust the zenith angle of the sun according to time of day.

        Parameters:
            time (float): Current time [days].
        """
        # When the diurnal cycle is disabled, use the constant zenith angle.
        if not self.diurnal_cycle:
            self.current_solar_angle = self.zenith_angle
            return

        # The solar angle is described by a sinusoidal curve that
        # oscillates around 90° (the horizon).
        self.current_solar_angle = ((self.zenith_angle + 90)
                                    + 90 * np.sin(2 * np.pi * time - np.pi / 2))

        # Zenith angles above 90° refer to nighttime. Set those angles to 90°.
        self.current_solar_angle = np.min((self.current_solar_angle, 90))


class PSRAD(Radiation):
    """Radiation model using the ICON PSRAD radiation scheme."""
    def _extract_psrad_args(self, atmosphere):
        """Returns tuple of mixing ratios to use with psrad.

        Paramteres:
            atmosphere (dict or pandas.DataFrame): Atmospheric atmosphere.

        Returns:
            tuple(ndarray): ndarrays in the order and unit to use with `psrad`:
                Z, P, T, x_vmr, ...
        """
        z = atmosphere['z'].values
        p = atmosphere['plev'].values / 100
        T = atmosphere['T'].values

        ret = [z, p, T]  # Z, P, T

        # Keep order as it is expected by PSRAD.
        required_gases = ['H2O', 'O3', 'CO2', 'N2O', 'CO', 'CH4']

        for gas in required_gases:
            if gas in atmosphere:
                ret.append(atmosphere[gas].values * 1e6)  # Append gas in ppm.
            else:
                ret.append(np.zeros(np.size(p)))

        return tuple(ret)

    @utils.PsradSymlinks()
    def get_heatingrates(self, atmosphere):
        """Returns the shortwave, longwave and net heatingrates.

        Parameters:
            atmosphere (conrad.atmosphere.Atmosphere): Atmosphere model.

        Returns:
            xarray.Dataset: Dataset containing for the simulated heating rates.
                The keys are 'sw_htngrt', 'lw_htngrt' and 'net_htngrt'.
        """
        from . import psrad
        self.psrad = psrad

        dmy_indices = np.asarray([0, 0, 0, 0])
        ic = dmy_indices.astype("int32") + 1
        c_lwc = np.asarray([0., 0., 0., 0.])
        c_iwc = np.asarray([0., 0., 0., 0.])
        c_frc = np.asarray([0., 0., 0., 0.])

        nlev = atmosphere['plev'].size

        # Extract surface properties.
        P_sfc = atmosphere.surface.pressure.values / 100
        T_sfc = atmosphere.surface.temperature.values[0]
        albedo = atmosphere.surface.albedo.values

        # Use the **current** solar angle as zenith angle for the simulation.
        zenith = self.current_solar_angle

        self.psrad.setup_single(nlev, len(ic), ic, c_lwc, c_iwc, c_frc, zenith,
                                albedo, P_sfc, T_sfc,
                                *self._extract_psrad_args(atmosphere))

        self.psrad.advance_lrtm()  # Longwave simulations.
        (lw_hr, lw_hr_clr, lw_flxd,
         lw_flxd_clr, lw_flxu, lw_flxu_clr) = self.psrad.get_lw_fluxes()

        self.psrad.advance_srtm()  # Shortwave simulations.

        (sw_hr, sw_hr_clr, sw_flxd,
         sw_flxd_clr, sw_flxu, sw_flxu_clr,
         vis_frc, par_dn, nir_dff, vis_diff,
         par_diff) = self.psrad.get_sw_fluxes()

        ret = Dataset({
            # General atmospheric properties.
            'z': atmosphere['z'],
            # Longwave fluxes and heatingrates.
            'lw_htngrt': (['time', 'plev'], lw_hr[:, :]),
            'lw_htngrt_clr': (['time', 'plev'], lw_hr_clr[:, :]),
            'lw_flxu': (['time', 'phlev'], lw_flxu[:, :]),
            'lw_flxd': (['time', 'phlev'], lw_flxd[:, :]),
            'lw_flxu_clr': (['time', 'phlev'], lw_flxu_clr[:, :]),
            'lw_flxd_clr': (['time', 'phlev'], lw_flxd_clr[:, :]),
            # Shortwave fluxes and heatingrates.
            # Note: The shortwave fluxes and heatingrates calculated by PSRAD
            # are **inverted**. Therefore, they are flipped to make the input
            # and output of this function consistent.
            'sw_htngrt': (['time', 'plev'], sw_hr[:, ::-1]),
            'sw_htngrt_clr': (['time', 'plev'], sw_hr_clr[:, ::-1]),
            'sw_flxu': (['time', 'phlev'], sw_flxu[:, ::-1]),
            'sw_flxd': (['time', 'phlev'], sw_flxd[:, ::-1]),
            'sw_flxu_clr': (['time', 'phlev'], sw_flxu_clr[:, ::-1]),
            'sw_flxd_clr': (['time', 'phlev'], sw_flxd_clr[:, ::-1]),
            # Net heatingrate.
            'net_htngrt': (['time', 'plev'], lw_hr[:, :] + sw_hr[:, ::-1]),
            # Radiation budget at top of the atmosphere (TOA).
            'toa': (['time'], (
                (sw_flxd[:, 0] + lw_flxd[:, -1])
                - (sw_flxu[:, 0] + lw_flxu[:, -1]))),
            },
            coords={
                'time': [0],
                'plev': atmosphere['plev'].values,
                'phlev': atmosphere['phlev'].values,
            }
            )

        append_description(ret)  # Append variable descriptions.

        return ret
