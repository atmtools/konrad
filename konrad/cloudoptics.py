"""Cloud optical properties from ECHAM."""
from os.path import (dirname, join)

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


class EchamCloudOptics:
    """Interface to interpolate cloud optical properties used in ECHAM."""
    def __init__(self):
        self.database = xr.open_dataset(
            join(dirname(__file__), 'data', 'ECHAM6_CldOptProps.nc')
        )

    def interp_ice_properties(self, particle_size=100., kind='linear'):
        x = self.database.re_crystal

        r = interp1d(
            x,
            self.database[f'co_albedo_crystal'],
            axis=1,
            kind=kind,
        )
        s = interp1d(
            x,
            self.database[f'asymmetry_factor_crystal'],
            axis=1,
            kind=kind,
        )
        t = interp1d(
            x,
            self.database[f'extinction_per_mass_crystal'],
            axis=1,
            kind=kind,
        )

        return (
            1 - r(particle_size)[16:],  # SW bands
            s(particle_size)[16:],
            t(particle_size)[16:],
            t(particle_size)[:16],  # LW bands
        )

    def interp_liquid_properties(self, particle_size=10., kind='linear'):
        x = self.database.re_droplet

        r = interp1d(
            x,
            self.database[f'co_albedo_droplet'],
            axis=1,
            kind='linear',
        )
        s = interp1d(
            x,
            self.database[f'asymmetry_factor_droplet'],
            axis=1,
            kind='linear',
        )
        t = interp1d(
            x,
            self.database[f'extinction_per_mass_droplet'],
            axis=1,
            kind='linear',
        )

        return (
            1 - r(particle_size)[16:],
            s(particle_size)[16:],
            t(particle_size)[16:],
            t(particle_size)[:16],
        )

    def get_cloud_properties(self, particle_size, water_path, phase='ice'):
        if phase == 'ice':
            ssa, asym, tau_sw, tau_lw = self.interp_ice_properties(
                particle_size)
        elif phase == 'liquid':
            ssa, asym, tau_sw, tau_lw = self.interp_liquid_properties(
                particle_size)
        else:
            raise ValueError(
                'Invalid phase. Allowed values are "ice" and "liquid".')

        cld_optics = xr.Dataset(
            coords={
                'num_shortwave_bands': np.arange(14),
                'num_longwave_bands': np.arange(16),
            },
        )

        cld_optics['single_scattering_albedo_due_to_cloud'] = (
            ('num_shortwave_bands',), ssa.ravel(),
        )

        cld_optics['cloud_asymmetry_parameter'] = (
            ('num_shortwave_bands',), asym.ravel(),
        )

        cld_optics['cloud_forward_scattering_fraction'] = (
            ('num_shortwave_bands',), asym.ravel() ** 2
        )

        cld_optics['shortwave_optical_thickness_due_to_cloud'] = (
            ('num_shortwave_bands',), water_path * tau_sw.ravel()
        )

        cld_optics['longwave_optical_thickness_due_to_cloud'] = (
            ('num_longwave_bands',), water_path * tau_lw.ravel()
        )

        return cld_optics
