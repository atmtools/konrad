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


# TODO: We need to find a clean way to handle the `numlevels`. Currently,
# it is easily possible to construct non-matching atmospehres and clouds.
class Cloud(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all cloud handlers.
    Default properties include a cloud area fraction equal to zero everywhere
    (ie no cloud)."""
    def __init__(self, z, num_lw_bands=16, num_sw_bands=14,
                 mass_water=0, mass_ice=0, cloud_fraction=0,
                 ice_particle_size=20, droplet_radius=10,
                 lw_optical_thickness=0, sw_optical_thickness=0,
                 forward_scattering_fraction=0, asymmetry_parameter=0.85,
                 single_scattering_albedo=0.9):
        """Create a cloud.

        Parameters:
            z (ndarray): array of height values [m]
            num_lw_bands (int): number of longwave bands, for RRTMG this is 16
            num_sw_bands (int): number of shortwave bands, for RRTMG this is 14
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
        numlevels = z.shape[0]

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

    def shift_cloud_profile(self, z, norm_new):
        """Shift the cloud area fraction according to a normalisation level.
        Maintain the depth of the cloud [m]
        Parameters:
            z (ndarray): height array [m]
            norm_new (int / float): normalisation height [m]
        """
        if self._f is None:
            normed_height = z - self._norm_level
            self._f = interp1d(
                normed_height,
                self.cloud_area_fraction_in_atmosphere_layer.values,
                fill_value='extrapolate',
            )

        # Move the cloud to the new normalisation level, if there is one.
        # Otherwise keep the cloud where it is.
        if norm_new is not np.nan:
            cloud_fraction_array = self._f(z - norm_new)
            self.cloud_area_fraction_in_atmosphere_layer = DataArray(
                cloud_fraction_array,
                dims=('mid_levels',),
                attrs={'units': 'dimensionless'})

    def scale_optical_thickness(self, z):
        """Conserve the total optical depth of the cloud.
        If we assume that the cloud remains at the same altitude and with the
        same thickness, then we also want to assume that the optical depth of
        the cloud remains the same.
        This is equivalent to conservation of cloud mass, as optical depth is
        proportional to density.

        Parameters:
            z (ndarray): height array
        """
        dz = np.hstack([z[0], np.diff(z)])  # TODO: is this a good approx?
        if self._longwave_optical_thickness_per_meter is None:  # first time
            self._longwave_optical_thickness_per_meter = \
                self.longwave_optical_thickness_due_to_cloud / dz
            self._shortwave_optical_thickness_per_meter = \
                self.shortwave_optical_thickness_due_to_cloud / dz

        self.longwave_optical_thickness_due_to_cloud = \
            dz * self._longwave_optical_thickness_per_meter
        self.shortwave_optical_thickness_due_to_cloud = \
            dz * self._shortwave_optical_thickness_per_meter

    @staticmethod
    def calculate_mass_cloud(cloud_fraction_array, z, density):
        dz = np.hstack((np.diff(z)[0], np.diff(z)))
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


class HighCloud(Cloud):
    def __init__(self, z, cloud_top=12000, depth=500,
                 cloud_fraction_in_cloud=1, ice_density=0.5,
                 ice_particle_size=20):
        """
        To be used with cloud_optical_properties='liquid_and_ice_clouds' in
        climt/RRTMG.

        Parameters:
            z (ndarray): altitude values [m]
            cloud_top (int): Cloud top height [m]
                This should be the convective top
            depth (int): Cloud depth / thickness of cloud [m]
            cloud_fraction_in_cloud (int / float): Cloud area fraction of the
                cloud for the model levels between cloud_top-depth and cloud_top
            ice_density (int / float): density of cloud ice [g m-3]
            ice_particle_size (int / float): Cloud ice particle size [microns]
        """
        cloud_fraction_array = np.zeros(z.shape)
        cloud_fraction_array[(z < cloud_top) & (
                z > cloud_top-depth)] = cloud_fraction_in_cloud
        cloud_fraction = DataArray(cloud_fraction_array, dims=('mid_levels',),
                                   attrs={'units': 'dimensionless'})

        mass_ice = self.calculate_mass_cloud(cloud_fraction_array, z,
                                             ice_density)

        super().__init__(
            z,
            cloud_fraction=cloud_fraction,
            mass_ice=mass_ice,
            ice_particle_size=ice_particle_size
        )

        self._ice_density = ice_density
        self._norm_level = cloud_top
        self._f = None

    def update_cloud_profile(self, atmosphere, convection, **kwargs):
        """ Keep the cloud attached to the convective top.
        """
        norm_new = convection.get('convective_top_height')[0]
        z = atmosphere.get('z', keepdims=False)
        self.shift_cloud_profile(z, norm_new)

        mass_ice = self.calculate_mass_cloud(
            self.cloud_area_fraction_in_atmosphere_layer.values,
            z,
            self._ice_density)
        self.mass_content_of_cloud_ice_in_atmosphere_layer = mass_ice


class LowCloud(Cloud):
    def __init__(self, z, cloud_top=2000, depth=1500,
                 cloud_fraction_in_cloud=1, water_density=0.5,
                 droplet_radius=10):
        """
        To be used with cloud_optical_properties='liquid_and_ice_clouds' in
        climt/RRTMG.

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

        mass_water = self.calculate_mass_cloud(cloud_fraction_array, z,
                                               water_density)

        super().__init__(
            z,
            cloud_fraction=cloud_fraction,
            mass_water=mass_water,
            droplet_radius=droplet_radius
        )

        self._water_density = water_density
        self._f = None
        self._norm_level = 0

    def update_cloud_profile(self, atmosphere, **kwargs):
        """ Keep the cloud fixed with height, in the planetary boundary layer.
        """
        z = atmosphere.get('z', keepdims=False)
        self.shift_cloud_profile(z, 0)

        mass = self.calculate_mass_cloud(
            self.cloud_area_fraction_in_atmosphere_layer.values,
            z,
            self._water_density)
        self.mass_content_of_cloud_liquid_water_in_atmosphere_layer = mass
