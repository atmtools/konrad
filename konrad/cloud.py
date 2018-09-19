# -*- coding: utf-8 -*-
"""This module contains classes for handling clouds."""
import abc
import logging

import numpy as np
from scipy.interpolate import interp1d
from sympl import DataArray

logger = logging.getLogger(__name__)


def get_p_data_array(values, units='kg m^-2', numlevels=200):
    """Return a DataArray of values."""
    if type(values) is DataArray:
        return values

    elif type(values) is int or type(values) is float:
        return DataArray(values*np.ones(numlevels,),
                         dims=('mid_levels',),
                         attrs={'units': units})

    raise TypeError(
            'Cloud variable input must be a single value or a sympl.DataArray')

def get_waveband_data_array(values, units='dimensionless', numbands=14,
                            numlevels=200, sw=True):
    """Return a DataArray of values."""
    if type(values) is DataArray:
        return values

    dims_bands = 'num_shortwave_bands' if sw else 'num_longwave_bands'

    if type(values) is int or type(values) is float:
        return DataArray(values*np.ones((numlevels, numbands)),
                         dims=('mid_levels', dims_bands),
                         attrs={'units': units})

    raise TypeError(
            'Cloud variable input must be a single value or a sympl.DataArray')


class Cloud(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all cloud handlers.
    Default properties include a cloud area fraction equal to zero everywhere
    (ie no cloud)."""
    def __init__(self, numlevels=200, num_lw_bands=16, num_sw_bands=14,
                 mass_water=0, mass_ice=0, cloud_fraction=0,
                 ice_particle_size=20, droplet_radius=10,
                 lw_optical_thickness=0, sw_optical_thickness=0,
                 forward_scattering_fraction=0.8, asymmetry_parameter=0.85,
                 single_scattering_albedo=0.9):
        """Create a cloud handler.

        Parameters:
            forward_scattering_fraction
        """
        self.mass_content_of_cloud_liquid_water_in_atmosphere_layer = \
                get_p_data_array(mass_water, numlevels=numlevels)

        self.mass_content_of_cloud_ice_in_atmosphere_layer = get_p_data_array(
                mass_ice, numlevels=numlevels)

        self.cloud_area_fraction_in_atmosphere_layer = get_p_data_array(
                cloud_fraction, numlevels=numlevels, units='dimensionless')

        self.cloud_ice_particle_size = get_p_data_array(ice_particle_size,
                                                        numlevels=numlevels,
                                                        units='micrometers')

        self.cloud_water_droplet_radius = get_p_data_array(droplet_radius,
                                                           numlevels=numlevels,
                                                           units='micrometers')

        self.longwave_optical_thickness_due_to_cloud = get_waveband_data_array(
                lw_optical_thickness, numlevels=numlevels,
                numbands=num_lw_bands, sw=False)

        self.cloud_forward_scattering_fraction = get_waveband_data_array(
                forward_scattering_fraction, numlevels=numlevels,
                numbands=num_sw_bands)

        self.cloud_asymmetry_parameter = get_waveband_data_array(
                asymmetry_parameter, numlevels=numlevels,
                numbands=num_sw_bands)

        self.shortwave_optical_thickness_due_to_cloud = \
                get_waveband_data_array(sw_optical_thickness,
                                        numlevels=numlevels,
                                        numbands=num_sw_bands)

        self.single_scattering_albedo_due_to_cloud = get_waveband_data_array(
                single_scattering_albedo, numlevels=numlevels,
                numbands=num_sw_bands)

    @abc.abstractmethod
    def update_cloud_profile(self, atmosphere, **kwargs):
        """Return the cloud parameters for the radiation scheme."""


class ClearSky(Cloud):
    def update_cloud_profile(self, *args, **kwargs):
        return


class HighCloud(Cloud):
    def __init__(self, z, cloud_top=12000, depth=500, area_fraction=1,
                 ice_density=0.5, lw_optical_thickness=10,
                 sw_optical_thickness=10,
                 forward_scattering_fraction=0.8, asymmetry_parameter=0.85,
                 single_scattering_albedo=0.9):
        """
        Parameters:
            z (ndarray): altitude values [m]
            cloud_top (int): Cloud top height [m]
            depth (int): Cloud depth / thickness of cloud [m]
            area_fraction (int/float): Area fraction covered by cloud
            ice_density (float): density of ice in the cloud [g m^-3]
            lw_optical_thickness (float): optical thickness of the cloud in the
                longwave [dimensionless]
            sw_optical_thickness (float): optical thickness of the cloud in the
                shortwave [dimensionless]
        """
        cloud_fraction_array = np.zeros(z.shape)
        cloud_fraction_array[(z < cloud_top) & (
                z > cloud_top-depth)] = area_fraction
        cloud_fraction = DataArray(cloud_fraction_array, dims=('mid_levels',),
                                   attrs={'units': 'dimensionless'})

        mass_ice = self.calculate_mass_cloud_ice(cloud_fraction_array, z,
                                                 ice_density=ice_density)

        super().__init__(cloud_fraction=cloud_fraction, mass_ice=mass_ice,
             lw_optical_thickness=lw_optical_thickness,
             sw_optical_thickness=sw_optical_thickness,
             forward_scattering_fraction=forward_scattering_fraction,
             asymmetry_parameter=asymmetry_parameter,
             single_scattering_albedo=single_scattering_albedo)

        self._ice_density = ice_density
        self._norm_level = cloud_top
        self._f = None

    def calculate_mass_cloud_ice(self, cloud_fraction_array, z, ice_density):
        dz = np.hstack((np.diff(z)[0], np.diff(z)))
        mass_ice_array = cloud_fraction_array * ice_density * 10 ** -3 * dz
        mass_ice = DataArray(mass_ice_array, dims=('mid_levels',),
                             attrs={'units': 'kg m^-2'})
        return mass_ice


    def update_cloud_profile(self, atmosphere, **kwargs):

        if self._f is None:
            normed_height = atmosphere.get_values(
                'z', keepdims=False)-self._norm_level
            self._f = interp1d(
                normed_height,
                self.cloud_area_fraction_in_atmosphere_layer.values,
                fill_value='extrapolate',
            )

        z = atmosphere.get_values('z', keepdims=False)
        norm_new = atmosphere.get_values('convective_top_height')

        cloud_fraction_array = self._f(z-norm_new)
        self.cloud_area_fraction_in_atmosphere_layer = DataArray(
            cloud_fraction_array,
            dims=('mid_levels',),
            attrs={'units': 'dimensionless'})

        mass_ice = self.calculate_mass_cloud_ice(cloud_fraction_array, z,
                                                 self._ice_density)
        self.mass_content_of_cloud_ice_in_atmosphere_layer = mass_ice
