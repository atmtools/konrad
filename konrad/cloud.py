# -*- coding: utf-8 -*-
"""This module contains classes for handling clouds."""
import abc
import logging

import numpy as np
from sympl import DataArray

logger = logging.getLogger(__name__)


class Cloud(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all cloud handlers."""
    def __init__(self, numlevels=200, num_lw_bands=16, num_sw_bands=14,
                 mass_water=None, mass_ice=None, cloud_fraction=None,
                 ice_particle_size=None, droplet_radius=None,
                 lw_optical_thickness=None, sw_optical_thickness=None,
                 forward_scattering_fraction=None, asymmetry_parameter=None,
                 single_scattering_albedo=None):
        """Create a cloud handler.

        Parameters:
            forward_scattering_fraction
        """

        if mass_water is not None:
            self.mass_content_of_cloud_liquid_water_in_atmosphere_layer = mass_water
        else:
            self.mass_content_of_cloud_liquid_water_in_atmosphere_layer = DataArray(
                    np.zeros(numlevels,),
                    dims=('mid_levels',),
                    attrs={'units': 'kg m^-2'})
        if mass_ice is not None:
            self.mass_content_of_cloud_ice_in_atmosphere_layer = mass_ice
        else:
            self.mass_content_of_cloud_ice_in_atmosphere_layer = DataArray(
                    np.zeros(numlevels,),
                    dims=('mid_levels',),
                    attrs={'units': 'kg m^-2'})
        if cloud_fraction is not None:
            self.cloud_area_fraction_in_atmosphere_layer = cloud_fraction
        else:
            self.cloud_area_fraction_in_atmosphere_layer = DataArray(
                    np.zeros(numlevels,),
                    dims=('mid_levels',),
                    attrs={'units': 'dimensionless'})
        if ice_particle_size is not None:
            #TODO: make case for entering a single value
            self.cloud_ice_particle_size = ice_particle_size
        else:
            self.cloud_ice_particle_size = DataArray(
                    20*np.ones(numlevels,),
                    dims=('mid_levels',),
                    attrs={'units': 'micrometers'})
        if droplet_radius is not None:
            #TODO: make case for entering a single value
            self.cloud_water_droplet_radius = droplet_radius
        else:
            self.cloud_water_droplet_radius = DataArray(
                    10*np.ones(numlevels,),
                    dims=('mid_levels',),
                    attrs={'units': 'micrometers'})

        if lw_optical_thickness is not None:
            self.longwave_optical_thickness_due_to_cloud = lw_optical_thickness
        else:
            self.longwave_optical_thickness_due_to_cloud = DataArray(
                np.zeros((1, num_lw_bands, 1, numlevels)),
                dims=('longitude', 'num_longwave_bands',
                      'latitude', 'mid_levels'),
                attrs={'units': 'dimensionless'})

        if forward_scattering_fraction is not None:
            self.cloud_forward_scattering_fraction = forward_scattering_fraction
        else:
            self.cloud_forward_scattering_fraction = DataArray(
                0.8*np.ones((1, num_sw_bands, 1, numlevels)),
                dims=('longitude', 'num_shortwave_bands',
                      'latitude', 'mid_levels'),
                attrs={'units': 'dimensionless'})
        if asymmetry_parameter is not None:
            self.cloud_asymmetry_parameter = asymmetry_parameter
        else:
            self.cloud_asymmetry_parameter = DataArray(
                0.85*np.ones((1, num_sw_bands, 1, numlevels)),
                dims=('longitude', 'num_shortwave_bands',
                      'latitude', 'mid_levels'),
                attrs={'units': 'dimensionless'})
        if sw_optical_thickness is not None:
            self.shortwave_optical_thickness_due_to_cloud = sw_optical_thickness
        else:
            self.shortwave_optical_thickness_due_to_cloud = DataArray(
                np.zeros((1, num_sw_bands, 1, numlevels)),
                dims=('longitude', 'num_shortwave_bands',
                      'latitude', 'mid_levels'),
                attrs={'units': 'dimensionless'})
        if single_scattering_albedo is not None:
            self.single_scattering_albedo_due_to_cloud = single_scattering_albedo
        else:
            self.single_scattering_albedo_due_to_cloud = DataArray(
                0.9*np.ones((1, num_sw_bands, 1, numlevels)),
                dims=('longitude', 'num_shortwave_bands',
                      'latitude', 'mid_levels'),
                attrs={'units': 'dimensionless'})

    @abc.abstractmethod
    def update_cloud_profile(self, **kwargs):
        """Return the cloud parameters for the radiation scheme."""


class ClearSky(Cloud):
    def __init__(self, numlevels=200):
        """Clear-sky. Set cloud area fraction to zero everywhere.

        Parameters:
            numlevels (float): Number of model levels.
        """
        cloud_fraction = DataArray(
                np.zeros(numlevels,),
                dims=('mid_levels',),
                attrs={'units': 'dimensionless'})
        super().__init__(cloud_fraction=cloud_fraction)
    def update_cloud_profile():
        return


class HighCloud(Cloud):
    def __init__(self, z):
        cloud_fraction_array = np.zeros(z.shape)
        cloud_fraction_array[(z < 12000) & (z > 11500)] = 1
        cloud_fraction = DataArray(
                cloud_fraction_array,
                dims=('mid_levels',),
                attrs={'units': 'dimensionless'})
        dz = np.hstack((np.diff(z)[0], np.diff(z)))
        mass_ice_array = cloud_fraction_array * 0.5 * 10**-3 * dz
        mass_ice = DataArray(
                mass_ice_array, dims=('mid_levels',),
                attrs={'units': 'kg m^-2'})

        super().__init__(cloud_fraction=cloud_fraction, mass_ice=mass_ice)

    def update_cloud_profile():
        return
