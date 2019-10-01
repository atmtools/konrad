"""This module contains a choice of clouds, which can be used either in the RCE
simulations or simply for radiative flux or heating rate calculations.
Depending on the choice of cloud, a certain set-up of the RRTMG radiation scheme
must be used.

**In an RCE simulation**

Create an instance of a cloud class, *e.g.* a :py:class:`DirectInputCloud`,
create an appropriate radiation model, and run an RCE simulation.
    >>> import konrad
    >>> cloudy_cloud = konrad.cloud.DirectInputCloud(
    >>>     numlevels=..., cloud_fraction=..., lw_optical_thickness=...,
    >>>     sw_optical_thickness=...)
    >>> rt = konrad.radiation.RRTMG(
    >>>     mcica=True, cloud_optical_properties='direct_input')
    >>> rce = konrad.RCE(atmosphere=..., cloud=cloudy_cloud, radiation=rt)
    >>> rce.run()

**Calculating radiative fluxes or heating rates**

Create an instance of a cloud class, *e.g.* a :py:class:`PhysicalCloud`,
create an appropriate radiation model and run radiative transfer.
    >>> import konrad
    >>> another_cloud = konrad.cloud.PhysicalCloud(
    >>>     numlevels=..., cloud_fraction=..., mass_ice=..., mass_water=...,
    >>>     ice_particle_size=..., droplet_radius=...)
    >>> rt = konrad.radiation.RRTMG(
    >>>     mcica=True, cloud_optical_properties='liquid_and_ice_clouds')
    >>> rt.calc_radiation(atmosphere=..., surface=..., cloud=another_cloud)

"""
import abc
import logging
import numbers

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


def get_rectangular_profile(z, value, ztop, depth):
    """Produce a rectangular profile, an array containing zeros and the value
    'value' corresponding to a certain height range.

    Parameters:
        z (ndarray): height
        value (int/float): non-zero value / thickness of rectangle
        ztop (int/float): height, indicating the top of the rectangle
        depth (int/float): height, indicating the depth of the rectangle
            ztop - depth gives the base of the rectangle
    """
    p = np.zeros(z.shape)
    inrectangle = np.logical_and(z < ztop, z > ztop - depth)
    p[inrectangle] = value

    return p


class Cloud(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all cloud handlers.
    Default properties include a cloud area fraction equal to zero everywhere
    (ie no cloud)."""

    #: number of longwave bands used in the radiation scheme
    num_longwave_bands = 16

    #: number of shortwave bands used in the radiation scheme
    num_shortwave_bands = 14

    def __init__(self, numlevels, cloud_fraction=0, mass_ice=0, mass_water=0,
                 ice_particle_size=20, droplet_radius=10,
                 lw_optical_thickness=0, sw_optical_thickness=0,
                 forward_scattering_fraction=0, asymmetry_parameter=0.85,
                 single_scattering_albedo=0.9,
                 rrtmg_cloud_optical_properties='liquid_and_ice_clouds',
                 rrtmg_cloud_ice_properties='ebert_curry_two',
                 ):
        """Create a cloud. Which of the input parameters are used and which
        ignored depends on the set-up of the radiation scheme.

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

            rrtmg_cloud_optical_properties (str):
                Choose how cloud properties are calculated by RRTMG.

                * :code:`direct_input`
                    Both cloud fraction and optical depth must be
                    input directly to the :py:mod:`konrad.cloud` instance.
                    Other cloud properties are irrelevant.

                * :code:`single_cloud_type`
                    Cloud fraction (1 or 0 at each level) and
                    cloud physical properties are required as input. Ice and
                    liquid water clouds are treated together, with a constant
                    value of cloud absorptivity. Not available with mcica.

                * :code:`liquid_and_ice_clouds`
                    Cloud fraction and cloud physical properties are required
                    as input. Ice and liquid clouds are treated separately.
                    Cloud optical depth is calculated from the cloud ice and
                    water particle sizes and the mass content of cloud and
                    water.

            rrtmg_cloud_ice_properties (str):
                Choose which method is used to calculate the cloud optical
                properties of ice clouds from their physical properties.

                * :code:`ebert_curry_one`
                * :code:`ebert_curry_two`
                * :code:`key_streamer_manual`
                * :code:`fu`
        """
        self.numlevels = numlevels

        self.mass_content_of_cloud_liquid_water_in_atmosphere_layer = self.get_p_data_array(
            mass_water)

        self.mass_content_of_cloud_ice_in_atmosphere_layer = self.get_p_data_array(
            mass_ice)

        self.cloud_area_fraction_in_atmosphere_layer = self.get_p_data_array(
            cloud_fraction, units='dimensionless')

        self.cloud_ice_particle_size = self.get_p_data_array(
            ice_particle_size, units='micrometers')

        self.cloud_water_droplet_radius = self.get_p_data_array(
            droplet_radius, units='micrometers')

        self.longwave_optical_thickness_due_to_cloud = self.get_waveband_data_array(
            lw_optical_thickness, sw=False)

        self.cloud_forward_scattering_fraction = self.get_waveband_data_array(
            forward_scattering_fraction)

        self.cloud_asymmetry_parameter = self.get_waveband_data_array(
            asymmetry_parameter)

        self.shortwave_optical_thickness_due_to_cloud = self.get_waveband_data_array(
            sw_optical_thickness)

        self.single_scattering_albedo_due_to_cloud = self.get_waveband_data_array(
            single_scattering_albedo)

        self._rrtmg_cloud_optical_properties = rrtmg_cloud_optical_properties
        self._rrtmg_cloud_ice_properties = rrtmg_cloud_ice_properties

    def get_p_data_array(self, values, units='kg m^-2'):
        """Return a DataArray of values."""
        if isinstance(values, DataArray):
            return values
        elif isinstance(values, np.ndarray):
            if values.shape != (self.numlevels,):
                raise ValueError(
                    'shape mismatch: Shape of cloud parameter input array '
                    f'{values.shape} is not compatible with number of model '
                    f'levels ({self.numlevels},).'
                )
        elif isinstance(values, numbers.Number):
            values = values * np.ones(self.numlevels,)
        else:
            raise TypeError(
                'Cloud variable input must be a single value, '
                '`numpy.ndarray` or a `sympl.DataArray`'
            )

        return DataArray(values, dims=('mid_levels',), attrs={'units': units})

    def get_waveband_data_array(self, values, units='dimensionless', sw=True):
        """Return a DataArray of values."""
        if isinstance(values, DataArray):
            return values

        if sw:
            dims = ('mid_levels', 'num_shortwave_bands')
            numbands = self.num_shortwave_bands
        else:
            dims = ('mid_levels', 'num_longwave_bands')
            numbands = self.num_longwave_bands

        if isinstance(values, numbers.Number):
            values = values * np.ones((self.numlevels, numbands))
        elif isinstance(values, np.ndarray):
            if values.shape == (self.numlevels,):
                values = np.repeat(
                    values[:, np.newaxis], numbands, axis=1)
            elif values.shape == (numbands,):
                values = np.repeat(
                    values[np.newaxis, :], self.numlevels, axis=0)
            elif not values.shape == (self.numlevels, numbands):
                raise ValueError(
                    f'shape mismatch: input array of shape {values.shape} '
                    'is not supported. Allowed shapes are: '
                    f'({self.numlevels},), ({numbands},), or '
                    f'({self.numlevels}, {numbands}).'
                )
        else:
            raise TypeError(
                'Cloud variable input must be a single value, '
                '`numpy.ndarray` or a `sympl.DataArray`'
            )

        return DataArray(values, dims=dims, attrs={'units': units})

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
    """No cloud.
    """
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
        """
        super().__init__(
            numlevels=numlevels,
            cloud_fraction=cloud_fraction,
            mass_ice=mass_ice,
            mass_water=mass_water,
            ice_particle_size=ice_particle_size,
            droplet_radius=droplet_radius,
            rrtmg_cloud_optical_properties='liquid_and_ice_clouds'
        )

    def update_cloud_profile(self, *args, **kwargs):
        """Keep the cloud fixed with pressure. """
        return


class DirectInputCloud(Cloud):
    """ To be used with cloud_optical_properties='direct_input' in climt/RRTMG.
    """
    def __init__(self, numlevels, cloud_fraction, lw_optical_thickness,
                 sw_optical_thickness, forward_scattering_fraction=0,
                 asymmetry_parameter=0.85, single_scattering_albedo=0.9,
                 norm_index=None):

        """Define a cloud based on properties that are directly used by the
        radiation scheme, namely cloud optical depth and scattering parameters.

        Parameters:
            numlevels (int): Number of atmospheric levels.
            cloud_fraction (float / ndarray / DataArray): cloud area fraction
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
            norm_index (int / None): model level index for coupling the cloud
        """

        super().__init__(
            numlevels=numlevels,
            cloud_fraction=cloud_fraction,
            lw_optical_thickness=lw_optical_thickness,
            sw_optical_thickness=sw_optical_thickness,
            forward_scattering_fraction=forward_scattering_fraction,
            asymmetry_parameter=asymmetry_parameter,
            single_scattering_albedo=single_scattering_albedo,
            rrtmg_cloud_optical_properties='direct_input'
        )

        self._norm_index = norm_index
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
            fill_value=0,
            bounds_error=False,
            axis=0,
        )
        return interpolation_f

    def shift_property(self, cloud_parameter, interpolation_f, norm_new):
        """Shift the cloud area fraction according to a normalisation level.

        Parameters:
            cloud_parameter (DataArray): cloud property to be shifted
            interpolation_f (scipy.interpolate.interpolate.interp1d):
                interpolation object calculated by interpolation_function
            norm_new (int): normalisation index [model level]

        Returns:
            DataArray: shifted cloud property
        """
        levels = np.arange(0, self.numlevels)
        if not np.isnan(norm_new):
            # Move the cloud to the new normalisation level, if there is one.
            cloud_parameter.values = interpolation_f(levels - norm_new)
        else:
            # Otherwise keep the cloud where it is.
            cloud_parameter.values = interpolation_f(levels - self._norm_index)

        return cloud_parameter

    def shift_cloud_profile(self, norm_new):
        if self._norm_index is None:
            self._norm_index = norm_new

        if self._interp_cldf is None:
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
            norm_new=convection.get('convective_top_index')[0]
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
        Parameters:
            numlevels (int): Number of atmospheric levels
            height_of_cloud (float/int): height at which the cloud is fixed [m]
            cloud_fraction (float / ndarray / DataArray): cloud area fraction
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
        super().__init__(
            numlevels=numlevels,
            cloud_fraction=cloud_fraction,
            lw_optical_thickness=lw_optical_thickness,
            sw_optical_thickness=sw_optical_thickness,
            forward_scattering_fraction=forward_scattering_fraction,
            asymmetry_parameter=asymmetry_parameter,
            single_scattering_albedo=single_scattering_albedo,
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
