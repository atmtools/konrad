"""This module contains a choice of clouds, which can be used either in the RCE
simulations or simply for radiative flux or heating rate calculations.
Depending on the choice of cloud, a certain set-up of the RRTMG radiation scheme
must be used.

**In an RCE simulation**

Create an instance of a cloud class, *e.g.* a :py:class:`DirectInputCloud`,
create an appropriate radiation model, and run an RCE simulation:

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
create an appropriate radiation model and run radiative transfer:

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

from konrad.component import Component
from konrad import utils
from konrad.cloudoptics import EchamCloudOptics

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
    'ConceptualCloud',
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


class Cloud(Component, metaclass=abc.ABCMeta):
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

        self.coords = {
            'mid_levels': np.arange(self.numlevels),
            'num_longwave_bands': np.arange(self.num_longwave_bands),
            'num_shortwave_bands': np.arange(self.num_shortwave_bands),
        }

        physical_props = {
            'mass_content_of_cloud_liquid_water_in_atmosphere_layer':
                (mass_water, 'kg m^-2'),
            'mass_content_of_cloud_ice_in_atmosphere_layer':
                (mass_ice, 'kg m^-2'),
            'cloud_area_fraction_in_atmosphere_layer':
                (cloud_fraction, 'dimensionless'),
            'cloud_ice_particle_size':
                (ice_particle_size, 'micrometers'),
            'cloud_water_droplet_radius':
                (droplet_radius, 'micrometers'),
        }

        for name, (var, unit) in physical_props.items():
            dataarray = self.get_p_data_array(var, units=unit)
            self[name] = dataarray.dims, dataarray

        cloud_optics = {
            'longwave_optical_thickness_due_to_cloud':
                (lw_optical_thickness, 'dimensionless', False),
            'cloud_forward_scattering_fraction':
                (forward_scattering_fraction, 'dimensionless', True),
            'cloud_asymmetry_parameter':
                (asymmetry_parameter, 'dimensionless', True),
            'shortwave_optical_thickness_due_to_cloud':
                (sw_optical_thickness, 'dimensionless', True),
            'single_scattering_albedo_due_to_cloud':
                (single_scattering_albedo, 'dimensionless', True),
        }

        for name, (var, unit, is_sw) in cloud_optics.items():
            dataarray = self.get_waveband_data_array(var, units=unit, sw=is_sw)
            self[name] = dataarray.dims, dataarray

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
    def update_cloud_profile(self, atmosphere, convection, radiation,
                             **kwargs):
        """Return the cloud parameters for the radiation scheme.
        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): atmosphere model
            convection (konrad.convection): convection scheme
            radiation (konrad.radiation): radiation scheme
        """

    def overcast(self):
        """Set cloud fraction in cloud layers to ``1`` (full overcast)."""
        cloud_fraction = self['cloud_area_fraction_in_atmosphere_layer'][:]
        cloud_mask = (cloud_fraction > 0).astype(float)

        self['cloud_area_fraction_in_atmosphere_layer'][:] = cloud_mask


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
        """Initialize a cloud component.

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
    direct_input_parameters = {
        'cloud_area_fraction_in_atmosphere_layer',
        'longwave_optical_thickness_due_to_cloud',
        'cloud_forward_scattering_fraction',
        'cloud_asymmetry_parameter',
        'shortwave_optical_thickness_due_to_cloud',
        'single_scattering_albedo_due_to_cloud',
    }

    def __init__(self, numlevels, cloud_fraction, lw_optical_thickness,
                 sw_optical_thickness, coupling='convective_top',
                 forward_scattering_fraction=0, asymmetry_parameter=0.85,
                 single_scattering_albedo=0.9, norm_index=None):

        """Define a cloud based on properties that are directly used by the
        radiation scheme, namely cloud optical depth and scattering parameters.

        Parameters:
            numlevels (int): Number of atmospheric levels.
            coupling (str): Mechanism with which the cloud is coupled to the
                atmospheric profile:

                    * 'convective_top': Coupling to the convective top
                    * 'freezing_level': Coupling to the freezing level
                    * 'subsidence_divergence': Coupling to the level of
                      maximum subsidence divergence
                    * 'pressure': Fixed at pressure (no coupling)

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
        self._interp_cache = {}
        self.coupling = coupling

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
                other['cloud_area_fraction_in_atmosphere_layer']
                > self['cloud_area_fraction_in_atmosphere_layer']
        )

        kwargs = {}
        for kwname, varname in name_map:
            arr = self[varname].values.copy()
            arr[other_is_bigger] = other[varname][other_is_bigger]
            kwargs[kwname] = arr

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

        for varname in self.direct_input_parameters:
            if varname not in self._interp_cache:
                self._interp_cache[varname] = self.interpolation_function(
                    cloud_parameter=self[varname])

            self[varname][:] = self.shift_property(
                cloud_parameter=self[varname],
                interpolation_f=self._interp_cache[varname],
                norm_new=norm_new,
            )

    def update_cloud_profile(self, atmosphere, convection, radiation,
                             **kwargs):
        """Keep the cloud profile fixed with model level (pressure). """
        if self.coupling == 'convective_top':
            self.shift_cloud_profile(
                norm_new=convection.get('convective_top_index')[0]
            )
        elif self.coupling == 'freezing_level':
            self.shift_cloud_profile(
                norm_new=atmosphere.get_triple_point_index(),
            )
        elif self.coupling == 'subsidence_divergence':
            Qr = radiation['net_htngrt_clr'][-1]
            self.shift_cloud_profile(
                norm_new=atmosphere.get_subsidence_convergence_max_index(Qr),
            )
        elif self.coupling == 'pressure':
            return
        else:
            raise ValueError(
                'The cloud class has been initialized with an invalid '
                'cloud coupling mechanism.'
            )


class HighCloud(DirectInputCloud):
    """Representation of a high-level cloud.

    High-level clouds are coupled to the convective top by default. Another
    reasonable option is a coupling to the level of maximum diabatic
    subsidence divergence (`"subsidence_divergence"`).
    """
    def __init__(self, *args, coupling='convective_top', **kwargs):
        super().__init__(*args, coupling=coupling, **kwargs)


class MidLevelCloud(DirectInputCloud):
    """Representation of a mid-level cloud.

    Mid-level clouds are coupled to the freezing level.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, coupling='freezing_level', **kwargs)


class LowCloud(DirectInputCloud):
    """Representation of a low-level cloud.

    Low-level clouds are fixed in pressure coordinates.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, coupling='pressure', **kwargs)


class ConceptualCloud(DirectInputCloud):
    def __init__(
        self,
        atmosphere,
        cloud_top,
        depth,
        cloud_fraction,
        water_path=100e-3,
        particle_size=100.,
        phase='ice',
        coupling='pressure'
    ):
        """Initialize a conceptual cloud.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.

            cloud_top (float): Pressure at cloud top [Pa].
            depth (float): Cloud depths in pressure units [Pa].
            cloud_fraction (float): Cloud fraction [0-1].
            water_path (float): Integrated water path [g m^-2].
            particle_size (float): Cloud particle size [microns].
            phase (str): Phase of cloud particles, either "ice" or "liquid".
            coupling (str): Mechanism with which the cloud top is coupled to
                the atmosphere profile:

                    * "pressure": Fixed at given pressure.
                    * "convective_top": Coupled to the convectio top.
                    * "freezing_level": Coupled to the freezing level.
                    * "subsidence_divergence: Coupled to the maximum subsidence
                      divergence.
                    * "temperature:TTT": Coupled to the level where the
                      temperature falls below `TTT` K.

        """
        super().__init__(
            numlevels=atmosphere['plev'].size,
            cloud_fraction=np.nan,
            lw_optical_thickness=np.nan,
            sw_optical_thickness=np.nan,
        )

        self["cloud_top"] = ("time",), np.array([cloud_top])
        self["cloud_top_temperature"] = ("time",), np.array([np.nan])
        self.depth = depth
        self.coupling = coupling

        self.cloud_fraction = cloud_fraction
        self.water_path = water_path
        self.particle_size = particle_size
        self.phase = phase

        self.update_cloud_profile(atmosphere)

    def get_cloud_optical_properties(self, water_content):
        cld_opt_props = EchamCloudOptics()

        return cld_opt_props.get_cloud_properties(
            self.particle_size, water_content, self.phase)

    @classmethod
    def from_atmosphere(cls, atmosphere, **kwargs):
        return cls(atmosphere['plev'].size, **kwargs)

    def update_cloud_top_plev(self, atmosphere, convection=None, radiation=None):
        """Determine cloud top pressure depending on coupling mechanism."""
        if self.coupling.lower() == 'pressure':
            return
        elif self.coupling.lower() == 'convective_top':
            if convection is not None:
                self.cloud_top = convection.get('convective_top_plev')[0]
        elif self.coupling.lower() == 'freezing_level':
            self["cloud_top"][:] = atmosphere.get_triple_point_plev()
            self["cloud_top"][:] -= self.depth / 2  # Center around freezing level
        elif self.coupling.lower() == 'subsidence_divergence':
            if radiation is not None:
                Qr = radiation['net_htngrt_clr'][-1]
                self["cloud_top"][:] = atmosphere.get_subsidence_convergence_max_plev(Qr)
        elif self.coupling.lower().startswith('temperature'):
            # Retrieve target temperature from keyword.
            threshold = float(self.coupling.split(":")[-1])

            # Because of the atmospheric temperature profile values around 220K
            # are ambiguous. Therefore, we are limiting the possible search
            # range to the troposphere
            cold_point = atmosphere.get_cold_point_plev()
            is_troposphere = atmosphere["plev"] > cold_point

            idx = np.abs(atmosphere["T"][-1, is_troposphere] - threshold).argmin()
            self["cloud_top"][:] = atmosphere["plev"][idx]
        else:
            raise ValueError(
                'The cloud class has been initialized with an invalid '
                'cloud coupling mechanism.'
            )

    def update_cloud_top_temperature(self, atmosphere):
        """Determine the cloud top temperature"""
        T = atmosphere["T"][-1]
        p = atmosphere["plev"]

        self["cloud_top_temperature"][:] = T[np.abs(p - self["cloud_top"]).argmin()]

    def update_cloud_profile(self, atmosphere, convection=None, radiation=None, **kwargs):
        """Update the cloud profile depending on the atmospheric state."""
        self.update_cloud_top_plev(atmosphere, convection, radiation)
        self.update_cloud_top_temperature(atmosphere)

        is_cloud = np.logical_and(
            atmosphere['plev'] > self["cloud_top"],
            atmosphere['plev'] < self["cloud_top"] + self.depth,
        ).astype(bool)

        self['cloud_area_fraction_in_atmosphere_layer'][:] = (
            self.cloud_fraction * is_cloud
        )

        water_content_per_Layer = self.water_path / np.sum(is_cloud)

        cloud_optics = self.get_cloud_optical_properties(
            water_content=water_content_per_Layer)

        for name in cloud_optics.data_vars:
            self[name][:, :] = 0
            self[name][is_cloud, :] = cloud_optics[name]


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
            self._clouds = np.asarray(args)

        self._superposition = None
        self.superpose()

        self.coords = self._superposition.coords

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError

        return getattr(self._superposition, name)

    def __getitem__(self, name):
        return self._superposition[name]

    def superpose(self):
        """Update the superposed cloud profile."""
        self._superposition = np.sum(self._clouds)

    @property
    def attrs(self):
        """Dictionary containing all attributes."""
        return self._superposition._attrs

    @property
    def data_vars(self):
        """Dictionary containing all data variables and their dimensions."""
        return self._superposition._data_vars

    @property
    def netcdf_subgroups(self):
        """Dynamically create a netCDF subgroup for each cloud."""
        return {f"cloud-{i}": cloud for i, cloud in enumerate( self._clouds)}

    def update_cloud_profile(self, *args, **kwargs):
        """Update every cloud in the cloud ensemble."""
        for cloud in self._clouds:
            cloud.update_cloud_profile(*args, **kwargs)

        self.superpose()

    def get_combinations(self):
        """Get all combinations of overlapping cloud layers."""
        if not all([isinstance(c, ConceptualCloud) for c in self._clouds]):
            raise TypeError(
                'Only `ConceptualCloud`s can be combined.'
            )

        bool_index, combined_weights = utils.calculate_combined_weights(
            weights=[cld.cloud_fraction for cld in self._clouds]
        )

        clouds = []
        for (i, p) in zip(bool_index, combined_weights):
            if not any(i):
                clouds.append(DirectInputCloud(
                    numlevels=self.coords['mid_levels'].size,
                    cloud_fraction=0.,
                    lw_optical_thickness=0.,
                    sw_optical_thickness=0.,
                ))
            else:
                composed_clouds = np.sum(self._clouds[i])
                composed_clouds.overcast()

                clouds.append(composed_clouds)

        return combined_weights, clouds
