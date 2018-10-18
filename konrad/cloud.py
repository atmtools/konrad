# -*- coding: utf-8 -*-
"""This module contains classes for handling clouds."""
import abc
import logging

import numpy as np
from scipy.interpolate import interp1d
from sympl import DataArray
from konrad.utils import dz_from_z

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
    num_longwave_bands = 16
    num_shortwave_bands = 14

    def __init__(self, z,
                 mass_water=0, mass_ice=0, cloud_fraction=0,
                 ice_particle_size=20, droplet_radius=10,
                 lw_optical_thickness=0, sw_optical_thickness=0,
                 forward_scattering_fraction=0, asymmetry_parameter=0.85,
                 single_scattering_albedo=0.9):
        """Create a cloud.

        Parameters:
            z (ndarray): array of height values [m]
            mass_water (float / DataArray): mass content of cloud liquid water
                [kg m-2]
            mass_ice (float / DataArray): mass content of cloud ice [kg m-2]
            cloud_fraction (float / DataArray): cloud area fraction
            ice_particle_size (float / DataArray): cloud ice particle size
                [micrometers]
            droplet_radius (float / DataArray): cloud water droplet radius
                [micrometers]
            lw_optical_thickness (float / DataArray): longwave optical
                thickness of the cloud
            sw_optical_thickness (float / DataArray): shortwave optical
                thickness of the cloud
            forward_scattering_fraction (float / DataArray): cloud forward
                scattering fraction (for the shortwave component of RRTMG)
                This is a scaling factor for the other shortwave parameters,
                if it is set to 0, no scaling is applied.
            asymmetry_parameter (float / DataArray): cloud asymmetry parameter
                (for the shortwave component of RRTMG)
            single_scattering_albedo (float / DataArray): single scattering
                albedo due to cloud (for the shortwave component of RRTMG)
        """
        numlevels = z.size

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
                numbands=self.num_longwave_bands, sw=False)

        self.cloud_forward_scattering_fraction = get_waveband_data_array(
                forward_scattering_fraction, numlevels=numlevels,
                numbands=self.num_shortwave_bands)

        self.cloud_asymmetry_parameter = get_waveband_data_array(
                asymmetry_parameter, numlevels=numlevels,
                numbands=self.num_shortwave_bands)

        self.shortwave_optical_thickness_due_to_cloud = \
            get_waveband_data_array(sw_optical_thickness,
                                    numlevels=numlevels,
                                    numbands=self.num_shortwave_bands)

        self.single_scattering_albedo_due_to_cloud = get_waveband_data_array(
                single_scattering_albedo, numlevels=numlevels,
                numbands=self.num_shortwave_bands)

    def interpolation_function(self, cloud_parameter, z):
        """ Calculate the interpolation function, to be used to keep the
        cloud at a constant thickness [m] and attached to a normalisation
        level (self._norm_level). A separate interpolation function is required
        for each cloud parameter that needs to be interpolated.

        Parameters:
            cloud_parameter (DataArray): property to be interpolation
            z (ndarray): height array, on which the interpolation is performed
        Returns:
            scipy.interpolate.interpolate.interp1d
        """
        normed_height = z - self._norm_level
        interpolation_f = interp1d(
            normed_height,
            cloud_parameter.values,
            axis=0,
            fill_value='extrapolate',
        )
        return interpolation_f

    def shift_property(self, cloud_parameter, interpolation_f, z, norm_new):
        """Shift the cloud area fraction according to a normalisation level.
        Maintain the thickness of the cloud [m].

        Parameters:
            cloud_parameter (DataArray): cloud property to be shifted
            interpolation_f (scipy.interpolate.interpolate.interp1d):
                interpolation object calculated by interpolation_function
            z (ndarray): height array [m]
            norm_new (int / float): normalisation height [m]

        Returns:
            DataArray: shifted cloud property
        """
        if norm_new is not np.nan:
            # Move the cloud to the new normalisation level, if there is one.
            cloud_parameter.values = interpolation_f(z - norm_new)
        else:
            # Otherwise keep the cloud where it is.
            cloud_parameter.values = interpolation_f(z - self._norm_level)

        return cloud_parameter

    def scale_optical_thickness(self, dz):
        """Conserve the total optical depth of the cloud.
        If we assume that the cloud remains at the same altitude and with the
        same thickness, then we also want to assume that the optical depth of
        the cloud remains the same.
        This is equivalent to conservation of cloud mass, as optical depth is
        proportional to density.

        Parameters:
            dz (ndarray): thickness of model levels [m]
        """
        dz_sw_array = np.repeat(dz[:, np.newaxis], self.num_shortwave_bands,
                                axis=1)
        dz_lw_array = np.repeat(dz[:, np.newaxis], self.num_longwave_bands,
                                axis=1)

        self.longwave_optical_thickness_due_to_cloud = DataArray(
            dz_lw_array * self._longwave_optical_thickness_per_meter,
            dims=('mid_levels', 'num_longwave_bands'),
            attrs={'units': 'dimensionless'}
        )
        self.shortwave_optical_thickness_due_to_cloud = DataArray(
            dz_sw_array * self._shortwave_optical_thickness_per_meter,
            dims=('mid_levels', 'num_shortwave_bands'),
            attrs={'units': 'dimensionless'}
        )

    @staticmethod
    def calculate_mass_cloud(cloud_fraction_array, dz, density):
        """Calculate the mass of the cloud on each model level given the cloud
        area fraction and the density within the cloud.

        Parameters:
            cloud_fraction_array (ndarray): cloud area fraction
            dz (ndarray): thickness of model levels [m]
            density (int / float): density of cloud mass [g m^-3]

        Returns:
            DataArray: mass of cloud on each model level [kg m^-2]
        """
        mass_array = cloud_fraction_array * density * 10 ** -3 * dz
        mass = DataArray(mass_array, dims=('mid_levels',),
                         attrs={'units': 'kg m^-2'})
        return mass

    @abc.abstractmethod
    def update_cloud_profile(self, atmosphere, convection, **kwargs):
        """Return the cloud parameters for the radiation scheme.
        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): atmosphere model
            convection (konrad.convection): convection scheme
        """


class ClearSky(Cloud):
    def update_cloud_profile(self, *args, **kwargs):
        return


class DirectInputCloud(Cloud):
    """ To be used with cloud_optical_properties='direct_input' in climt/RRTMG.
    """
    def __init__(self, z, cloud_fraction,
                 lw_optical_thickness, sw_optical_thickness,
                 forward_scattering_fraction=0, asymmetry_parameter=0.85,
                 single_scattering_albedo=0.9):

        super().__init__(
            z,
            cloud_fraction=cloud_fraction,
            lw_optical_thickness=lw_optical_thickness,
            sw_optical_thickness=sw_optical_thickness,
            forward_scattering_fraction=forward_scattering_fraction,
            asymmetry_parameter=asymmetry_parameter,
            single_scattering_albedo=single_scattering_albedo
        )

        dz = dz_from_z(z)
        dz_sw_array = np.repeat(dz[:, np.newaxis], self.num_shortwave_bands,
                                axis=1)
        dz_lw_array = np.repeat(dz[:, np.newaxis], self.num_longwave_bands,
                                axis=1)
        self._longwave_optical_thickness_per_meter = \
            self.longwave_optical_thickness_due_to_cloud / dz_lw_array
        self._shortwave_optical_thickness_per_meter = \
            self.shortwave_optical_thickness_due_to_cloud / dz_sw_array

        self._norm_level = 0
        self._f = self.interpolation_function(
            self.cloud_area_fraction_in_atmosphere_layer, z)
        self._interpolation_sw = self.interpolation_function(
            self._shortwave_optical_thickness_per_meter, z)
        self._interpolation_lw = self.interpolation_function(
            self._longwave_optical_thickness_per_meter, z)

    def update_cloud_profile(self, atmosphere, **kwargs):
        """ Keep the cloud profile fixed with height.
        """
        z = atmosphere.get('z', keepdims=False)
        dz = dz_from_z(z)

        self.cloud_area_fraction_in_atmosphere_layer = self.shift_property(
            self.cloud_area_fraction_in_atmosphere_layer, self._f, z, 0)

        self._shortwave_optical_thickness_per_meter = self.shift_property(
            self._shortwave_optical_thickness_per_meter,
            self._interpolation_sw, z, 0)
        self._longwave_optical_thickness_per_meter = self.shift_property(
            self._longwave_optical_thickness_per_meter,
            self._interpolation_lw, z, 0)

        self.scale_optical_thickness(dz)


class HighCloud(Cloud):
    """ To be used with cloud_optical_properties='liquid_and_ice_clouds' in
    climt/RRTMG."""
    def __init__(self, z, cloud_top=12000, depth=500,
                 cloud_fraction_in_cloud=1, ice_density=0.5,
                 ice_particle_size=20):
        """
        Parameters:
            z (ndarray): altitude values [m]
            cloud_top (int): Cloud top height [m]
                This should be the convective top
            depth (int): Cloud depth / thickness of cloud [m]
            cloud_fraction_in_cloud (int / float): Cloud area fraction of the
                cloud for the model levels between cloud_top - depth and
                cloud_top
            ice_density (int / float): density of cloud ice [g m-3]
            ice_particle_size (int / float): Cloud ice particle size [microns]
        """
        cloud_fraction_array = np.zeros(z.shape)
        cloud_fraction_array[(z < cloud_top) & (
                z > cloud_top-depth)] = cloud_fraction_in_cloud
        cloud_fraction = DataArray(cloud_fraction_array, dims=('mid_levels',),
                                   attrs={'units': 'dimensionless'})

        mass_ice = self.calculate_mass_cloud(cloud_fraction_array, dz_from_z(z),
                                             ice_density)

        super().__init__(
            z,
            cloud_fraction=cloud_fraction,
            mass_ice=mass_ice,
            ice_particle_size=ice_particle_size
        )

        self._ice_density = ice_density
        self._norm_level = cloud_top
        self._f = self.interpolation_function(
            self.cloud_area_fraction_in_atmosphere_layer, z)

    def update_cloud_profile(self, atmosphere, convection, **kwargs):
        """ Keep the cloud attached to the convective top.
        """
        norm_new = convection.get('convective_top_height')[0]
        z = atmosphere.get('z', keepdims=False)
        dz = dz_from_z(z)

        self.cloud_area_fraction_in_atmosphere_layer = self.shift_property(
            self.cloud_area_fraction_in_atmosphere_layer, self._f, z, norm_new)

        mass_ice = self.calculate_mass_cloud(
            self.cloud_area_fraction_in_atmosphere_layer.values,
            dz,
            self._ice_density)
        self.mass_content_of_cloud_ice_in_atmosphere_layer = mass_ice


class LowCloud(Cloud):
    """ To be used with cloud_optical_properties='liquid_and_ice_clouds' in
    climt/RRTMG."""
    def __init__(self, z, cloud_top=2000, depth=1500,
                 cloud_fraction_in_cloud=1, water_density=0.5,
                 droplet_radius=10):
        """
        Parameters:
            z (ndarray): altitude values [m]
            cloud_top (int): Cloud top height [m]
            depth (int): Cloud depth / thickness of cloud [m]
            cloud_fraction_in_cloud (int): Cloud area fraction of the cloud for
                the model levels between cloud_top-depth and cloud_top
            water_density (int / float): density of cloud water [g m-3]
            droplet_radius (int / float): Cloud water droplet radius [microns]
        """
        cloud_fraction_array = np.zeros(z.shape)
        cloud_fraction_array[(z < cloud_top) & (
                z > cloud_top-depth)] = cloud_fraction_in_cloud
        cloud_fraction = DataArray(cloud_fraction_array, dims=('mid_levels',),
                                   attrs={'units': 'dimensionless'})

        mass_water = self.calculate_mass_cloud(cloud_fraction_array,
                                               dz_from_z(z), water_density)

        super().__init__(
            z,
            cloud_fraction=cloud_fraction,
            mass_water=mass_water,
            droplet_radius=droplet_radius
        )

        self._water_density = water_density
        self._f = self.interpolation_function(
            self.cloud_area_fraction_in_atmosphere_layer, z)
        self._norm_level = 0

    def update_cloud_profile(self, atmosphere, **kwargs):
        """ Keep the cloud fixed with height, in the planetary boundary layer.
        """
        z = atmosphere.get('z', keepdims=False)
        dz = dz_from_z(z)

        self.cloud_area_fraction_in_atmosphere_layer = self.shift_property(
            self.cloud_area_fraction_in_atmosphere_layer, self._f, z, 0)

        mass = self.calculate_mass_cloud(
            self.cloud_area_fraction_in_atmosphere_layer.values,
            dz,
            self._water_density)
        self.mass_content_of_cloud_liquid_water_in_atmosphere_layer = mass
