"""This module contains classes for handling clouds."""
import abc
import logging

import numpy as np
from scipy.interpolate import interp1d
from sympl import DataArray

logger = logging.getLogger(__name__)


__all__ = [
    'get_rectangular_profile',
    'Cloud',
    'ClearSky',
    'DirectInputCloud',
    'PhysicalCloud',
    'HighCloud',
    'MidLevelCloud',
    'LowCloud',
    'CloudEnsemble',
]


# TODO: Make this a staticmethod of the ``Cloud`` class?
def get_p_data_array(values, units='kg m^-2', numlevels=200):
    """Return a DataArray of values."""
    if isinstance(values, DataArray):
        return values

    elif isinstance(values, np.ndarray):
        if values.shape == (numlevels,):
            return DataArray(values, dims=('mid_levels',),
                             attrs={'units': units})
        else:
            raise ValueError(
                'Cloud parameter input array is not the right size'
                ' for the number of model levels.')

    elif isinstance(values, (int, float)):
        return DataArray(values * np.ones(numlevels, ),
                         dims=('mid_levels',),
                         attrs={'units': units})

    raise TypeError(
        'Cloud variable input must be a single value, `numpy.ndarray` or a '
        '`sympl.DataArray`')


# TODO: Make this a staticmethod of the ``Cloud`` class?
def get_waveband_data_array(values, units='dimensionless', numlevels=200,
                            sw=True):
    """Return a DataArray of values."""
    if isinstance(values, DataArray):
        return values

    if sw:
        dims_bands = 'num_shortwave_bands'
        numbands = 14
    else:
        dims_bands = 'num_longwave_bands'
        numbands = 16

    if isinstance(values, (int, float)):
        return DataArray(values * np.ones((numlevels, numbands)),
                         dims=('mid_levels', dims_bands),
                         attrs={'units': units})

    elif isinstance(values, np.ndarray):
        if values.shape == (numlevels,):
            return DataArray(
                np.repeat(values[:, np.newaxis], numbands, axis=1),
                dims=('mid_levels', dims_bands),
                attrs={'units': units},
            )
        elif values.shape == (numlevels, numbands):
            return DataArray(
                values,
                dims=('mid_levels', dims_bands),
                attrs={'units': units},
            )

    raise TypeError(
        'Cloud variable input must be a single value, `numpy.ndarray` or a '
        '`sympl.DataArray`')


def get_rectangular_profile(z, value, ztop, depth):
    p = np.zeros(z.shape)
    inrectangle = np.logical_and(z < ztop, z > ztop - depth)
    p[inrectangle] = value

    return p


class Cloud(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all cloud handlers.
    Default properties include a cloud area fraction equal to zero everywhere
    (ie no cloud)."""
    num_longwave_bands = 16
    num_shortwave_bands = 14

    def __init__(self, numlevels, cloud_fraction=0, mass_ice=0, mass_water=0,
                 ice_particle_size=20, droplet_radius=10,
                 lw_optical_thickness=0, sw_optical_thickness=0,
                 forward_scattering_fraction=0, asymmetry_parameter=0.85,
                 single_scattering_albedo=0.9):
        """Create a cloud.

        Parameters:
            numlevels (int): Number of atmospheric levels.
            cloud_fraction (float / ndarray / DataArray): cloud area fraction
            mass_ice (float / ndarray / DataArray): mass content of cloud ice
                [kg m-2]
            mass_water (float / ndarray / DataArray): mass content of cloud
                liquid water [kg m-2]
            ice_particle_size (float / ndarray / DataArray): cloud ice particle
                size [micrometers]
            droplet_radius (float / ndarray / DataArray): cloud water droplet
                radius [micrometers]
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
        self.numlevels = numlevels

        self.mass_content_of_cloud_liquid_water_in_atmosphere_layer = \
            get_p_data_array(mass_water, numlevels=self.numlevels)

        self.mass_content_of_cloud_ice_in_atmosphere_layer = get_p_data_array(
            mass_ice, numlevels=self.numlevels)

        self.cloud_area_fraction_in_atmosphere_layer = get_p_data_array(
            cloud_fraction, numlevels=self.numlevels, units='dimensionless')

        self.cloud_ice_particle_size = get_p_data_array(
            ice_particle_size, numlevels=self.numlevels, units='micrometers')

        self.cloud_water_droplet_radius = get_p_data_array(
            droplet_radius, numlevels=self.numlevels, units='micrometers')

        self.longwave_optical_thickness_due_to_cloud = get_waveband_data_array(
            lw_optical_thickness, numlevels=self.numlevels, sw=False)

        self.cloud_forward_scattering_fraction = get_waveband_data_array(
            forward_scattering_fraction, numlevels=self.numlevels)

        self.cloud_asymmetry_parameter = get_waveband_data_array(
            asymmetry_parameter,numlevels=self.numlevels)

        self.shortwave_optical_thickness_due_to_cloud = \
            get_waveband_data_array(sw_optical_thickness,
                                    numlevels=self.numlevels)

        self.single_scattering_albedo_due_to_cloud = get_waveband_data_array(
            single_scattering_albedo, numlevels=self.numlevels)

    @classmethod
    def from_atmosphere(cls, atmosphere, **kwargs):
        """Initialize a cloud component matching the given atmosphere.

        Parameters:
            atmosphere (``konrad.atmosphere.Atmosphere``):
                Atmosphere component.

        """
        return cls(numlevels=atmosphere['plev'].size, **kwargs)

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


class PhysicalCloud(Cloud):
    """Define a cloud based on physical properties.

    The physical properties are cloud ice and liquid mass (per model level)
    and particle size.  To be used with
    cloud_optical_properties='liquid_and_ice_clouds' in climt/RRTMG.
    """
    def __init__(self, numlevels, cloud_fraction, mass_water, mass_ice,
                 ice_particle_size, droplet_radius):
        """
        Parameters:
        z (ndarray): an array with the size of the model levels
        cloud_fraction (float / ndarray / DataArray): cloud area fraction
        mass_ice (float / ndarray / DataArray): mass content of cloud ice
            [kg m-2]
        mass_water (float / ndarray / DataArray): mass content of cloud
            liquid water [kg m-2]
        ice_particle_size (float / ndarray / DataArray): cloud ice particle
            size [micrometers]
        droplet_radius (float / ndarray / DataArray): cloud water droplet
            radius [micrometers]
        """
        super().__init__(
            numlevels=numlevels,
            cloud_fraction=cloud_fraction,
            mass_ice=mass_ice,
            mass_water=mass_water,
            ice_particle_size=ice_particle_size,
            droplet_radius=droplet_radius
        )

    def update_cloud_profile(self, *args, **kwargs):
        """Keep the cloud fixed with pressure. """
        return


class DirectInputCloud(Cloud):
    """ To be used with cloud_optical_properties='direct_input' in climt/RRTMG.
    """
    def __init__(self, numlevels, cloud_fraction, lw_optical_thickness,
                 sw_optical_thickness, forward_scattering_fraction=0,
                 asymmetry_parameter=0.85, single_scattering_albedo=0.9):

        super().__init__(
            numlevels=numlevels,
            cloud_fraction=cloud_fraction,
            lw_optical_thickness=lw_optical_thickness,
            sw_optical_thickness=sw_optical_thickness,
            forward_scattering_fraction=forward_scattering_fraction,
            asymmetry_parameter=asymmetry_parameter,
            single_scattering_albedo=single_scattering_albedo
        )

        self._norm_index = None
        self._interp_cldf = None
        self._interp_sw = None
        self._interp_lw = None

    def __add__(self, other):
        """Define the superposition of two clouds in a layer."""
        name_map = (
            ('cloud_fraction',
             'cloud_area_fraction_in_atmosphere_layer'),
            ('lw_optical_thickness',
             'longwave_optical_thickness_due_to_cloud'),
            ('sw_optical_thickness',
             'shortwave_optical_thickness_due_to_cloud'),
            ('forward_scattering_fraction',
             'cloud_forward_scattering_fraction'),
            ('asymmetry_parameter',
             'cloud_asymmetry_parameter'),
            ('single_scattering_albedo',
             'single_scattering_albedo_due_to_cloud'),
        )

        # The superposition of two clouds is implemented following a
        # "The winner takes it all"-approach:
        # For each cloud layer, the properties of the bigger cloud (in terms
        # of cloud fraction) is used.
        other_is_bigger = (
                other.cloud_area_fraction_in_atmosphere_layer
                > self.cloud_area_fraction_in_atmosphere_layer
        )

        kwargs = {}
        for kw_name, attr_name in name_map:
            arr = getattr(self, attr_name).values.copy()
            arr[other_is_bigger] = getattr(other, attr_name)[other_is_bigger]
            kwargs[kw_name] = arr

        summed_cloud = DirectInputCloud(numlevels=self.numlevels, **kwargs)

        return summed_cloud

    def interpolation_function(self, cloud_parameter):
        """ Calculate the interpolation function, to be used to maintain the
        cloud optical properties and keep the cloud attached to a normalisation
        level (self._norm_index). A separate interpolation function is required
        for each cloud parameter that needs to be interpolated.

        Parameters:
            cloud_parameter (DataArray): property to be interpolation
        Returns:
            scipy.interpolate.interpolate.interp1d
        """
        normed_levels = np.arange(0, self.numlevels) - self._norm_index

        interpolation_f = interp1d(
            normed_levels,
            cloud_parameter.values,
            fill_value='extrapolate',
            axis=0,
        )
        return interpolation_f

    def shift_property(self, cloud_parameter, interpolation_f, norm_new):
        """Shift the cloud area fraction according to a normalisation level.

        Parameters:
            cloud_parameter (DataArray): cloud property to be shifted
            interpolation_f (scipy.interpolate.interpolate.interp1d):
                interpolation object calculated by interpolation_function
            norm_new (int / float): normalisation index [model level]

        Returns:
            DataArray: shifted cloud property
        """
        levels = np.arange(0, self.numlevels)
        if norm_new is not np.nan:
            # Move the cloud to the new normalisation level, if there is one.
            cloud_parameter.values = interpolation_f(levels - norm_new)
        else:
            # Otherwise keep the cloud where it is.
            cloud_parameter.values = interpolation_f(levels - self._norm_index)

        return cloud_parameter

    def shift_cloud_profile(self, norm_new):
        if self._norm_index is None:
            self._norm_index = norm_new

            self._interp_cldf = self.interpolation_function(
                cloud_parameter=self.cloud_area_fraction_in_atmosphere_layer,
            )

            self._interp_sw = self.interpolation_function(
                cloud_parameter=self.shortwave_optical_thickness_due_to_cloud,
            )

            self._interp_lw = self.interpolation_function(
                cloud_parameter=self.longwave_optical_thickness_due_to_cloud,
            )

        self.cloud_area_fraction_in_atmosphere_layer = self.shift_property(
            cloud_parameter=self.cloud_area_fraction_in_atmosphere_layer,
            interpolation_f=self._interp_cldf,
            norm_new=norm_new,
        )

        self.shortwave_optical_thickness_due_to_cloud = self.shift_property(
            cloud_parameter=self.shortwave_optical_thickness_due_to_cloud,
            interpolation_f=self._interp_sw,
            norm_new=norm_new,
        )

        self.longwave_optical_thickness_due_to_cloud = self.shift_property(
            cloud_parameter=self.longwave_optical_thickness_due_to_cloud,
            interpolation_f=self._interp_lw,
            norm_new=norm_new,
        )

    def update_cloud_profile(self, atmosphere, convection, **kwargs):
        """Keep the cloud profile fixed with model level (pressure). """
        return


class HighCloud(DirectInputCloud):
    """Representation of a high-level cloud."""
    def update_cloud_profile(self, atmosphere, convection, **kwargs):
        """Keep the cloud attached to the convective top. """
        self.shift_cloud_profile(
            norm_new=convection.get('convective_top_index')[0],
        )


class MidLevelCloud(DirectInputCloud):
    """Representation of a mid-level cloud."""
    def update_cloud_profile(self, atmosphere, convection, **kwargs):
        """Keep the cloud attached to the freezing point. """
        self.shift_cloud_profile(
            norm_new=atmosphere.get_triple_point_index(),
        )


class LowCloud(DirectInputCloud):
    """Representation of a low-level cloud.
    Fixed at the top of the planetary boundary layer."""
    def __init__(self, numlevels, height_of_cloud, cloud_fraction,
                 lw_optical_thickness, sw_optical_thickness,
                 forward_scattering_fraction=0, asymmetry_parameter=0.85,
                 single_scattering_albedo=0.9):
        """
        Additional parameters to parent class:
            height_of_cloud (float/int): height at which the cloud is fixed [m]
        """
        super().__init__(
            numlevels=numlevels,
            cloud_fraction=cloud_fraction,
            lw_optical_thickness=lw_optical_thickness,
            sw_optical_thickness=sw_optical_thickness,
            forward_scattering_fraction=forward_scattering_fraction,
            asymmetry_parameter=asymmetry_parameter,
            single_scattering_albedo=single_scattering_albedo
        )

        self._z_of_cloud = height_of_cloud

    def update_cloud_profile(self, atmosphere, convection, **kwargs):
        """Keep the cloud at a fixed height. """
        z = atmosphere.get('z')[-1]
        index = int(np.argmax(z > self._z_of_cloud))  # model level above
        height_array = np.array([z[index-1], z[index]])
        index_array = np.array([index-1, index])  # model level bounds
        # Interpolate between two indices for model levels to be closer to that
        # corresponding to the height at which the cloud should be fixed.
        norm_new = interp1d(height_array, index_array)(self._z_of_cloud)

        self.shift_cloud_profile(
            norm_new=norm_new
        )


class CloudEnsemble(DirectInputCloud):
    """Wrapper to combine several clouds into a cloud ensemble.

    Warning: For now, overlapping clouds are handled very poorly!

    A cloud ensemble can consist of an arbitrary number of clouds.
    After its initialization it is handled like a normal `Cloud`:

    >>> cloud1 = HighCloud(...)
    >>> cloud2 = LowCloud(...)
    >>> cloud_ensemble = CloudEnsemble(cloud1, cloud2)
    >>> cloud_ensemble.cloud_area_fraction_in_atmosphere_layer

    """
    def __init__(self, *args):
        if not all([isinstance(a, DirectInputCloud) for a in args]):
            raise ValueError(
                'Only `DirectInputCloud`s can be combined in an ensemble.')
        else:
            self.clouds = args

        self._superposition = None
        self.superpose()

    def __getattr__(self, name):
        return getattr(self._superposition, name)

    def __getitem__(self, name):
        return getattr(self, name)

    def superpose(self):
        """Update the superposed cloud profile."""
        self._superposition = np.sum(self.clouds)

    def update_cloud_profile(self, *args, **kwargs):
        """Update every cloud in the cloud ensemble."""
        for cloud in self.clouds:
            cloud.update_cloud_profile(*args, **kwargs)

        self.superpose()
