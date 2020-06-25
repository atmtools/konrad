"""Define an interface for the RRTMG radiation scheme (through CliMT). """
from copy import deepcopy
import numpy as np
import datetime
from sympl import DataArray
from typhon.physics import vmr2specific_humidity
import climt
import logging

from .radiation import Radiation
from konrad.cloud import ClearSky, CloudEnsemble

__all__ = [
    'RRTMG',
]


class RRTMG(Radiation):
    """RRTMG radiation scheme using the CliMT python wrapper."""

    def __init__(self, *args, solar_constant=551.58, mcica=False,
                 **kwargs):
        """
        Parameters:
            zenith_angle (float): angle of the Sun [degrees].
                In konrad, with no diurnal cycle, this should represent a
                diurnal mean zenith angle. With a diurnal cycle, this
                represents latitude.  For single radiation calculations, this
                is the angle to the Sun at the time and location of interest.

            bias (dict-like): include bias corrections to the fluxes and/or
                heating rates

            solar_constant (int): [W m^-2]

            mcica (bool):

                * :code:`False`
                    use the nomcica version of RRTMG (clear-sky or overcast)

                * :code:`True`
                    use the mcica version of RRTMG (needed for partly cloudy
                    skies)

        """
        super().__init__(*args, **kwargs)
        self._state_lw = None
        self._state_sw = None

        self._rad_lw = None
        self._rad_sw = None

        self._is_mcica = mcica

        # These are set in the first call and depend on properties set in the
        # cloud class or instance.
        self._cloud_optical_properties = None
        self._cloud_ice_properties = None

        self.solar_constant = solar_constant

    def init_radiative_state(self, atmosphere, surface):

        climt.set_constants_from_dict({"stellar_irradiance": {
                "value": self.solar_constant, "units": 'W m^-2'}})
        if self._is_mcica:
            overlap = 'maximum_random'
        else:
            overlap = 'random'
        self._rad_lw = climt.RRTMGLongwave(
            cloud_optical_properties=self._cloud_optical_properties,
            cloud_ice_properties=self._cloud_ice_properties,
            cloud_liquid_water_properties='radius_dependent_absorption',
            cloud_overlap_method=overlap,
            mcica=self._is_mcica)
        self._rad_sw = climt.RRTMGShortwave(
            ignore_day_of_year=True,
            cloud_optical_properties=self._cloud_optical_properties,
            cloud_ice_properties=self._cloud_ice_properties,
            cloud_liquid_water_properties='radius_dependent_absorption',
            cloud_overlap_method=overlap,
            mcica=self._is_mcica)
        state_lw = {}
        state_sw = {}

        plev = atmosphere['plev']
        phlev = atmosphere['phlev']
        numlevels = len(plev)

        for state0 in state_lw, state_sw:
            state0['mid_levels'] = DataArray(
                    np.arange(0, numlevels),
                    dims=('mid_levels',),
                    attrs={'label': 'mid_levels'})

            state0['interface_levels'] = DataArray(
                    np.arange(0, numlevels+1),
                    dims=('interface_levels',),
                    attrs={'label': 'interface_levels'})

            state0['air_pressure'] = DataArray(
                    plev,
                    dims=('mid_levels',),
                    attrs={'units': 'Pa'})

            state0['air_pressure_on_interface_levels'] = DataArray(
                    phlev,
                    dims=('interface_levels',),
                    attrs={'units': 'Pa'})

            state0['time'] = datetime.datetime(2000, 1, 1)

        ### Aerosols ###
        # TODO: Should all the aerosol values be zero?!
        # Longwave specific
        num_lw_bands = self._rad_lw.num_longwave_bands

        state_lw['longwave_optical_thickness_due_to_aerosol'] = DataArray(
                np.zeros((num_lw_bands, numlevels)),
                dims=('num_longwave_bands', 'mid_levels'),
                attrs={'units': 'dimensionless'})
        state_lw['surface_longwave_emissivity'] = DataArray(
            surface.longwave_emissivity * np.ones((num_lw_bands,)),
            dims=('num_longwave_bands'),
            attrs={'units': 'dimensionless'})

        # Shortwave specific changes
        num_sw_bands = self._rad_sw.num_shortwave_bands

        num_aerosols = self._rad_sw.num_ecmwf_aerosols
        state_sw['aerosol_optical_depth_at_55_micron'] = DataArray(
                np.zeros((num_aerosols, numlevels)),
                dims=('num_ecmwf_aerosols', 'mid_levels'),
                attrs={'units': 'dimensionless'})
        state_sw['solar_cycle_fraction'] = DataArray(
            0, attrs={'units': 'dimensionless'})
        state_sw['flux_adjustment_for_earth_sun_distance'] = DataArray(
            1, attrs={'units': 'dimensionless'})

        for surface_albedo in ['surface_albedo_for_diffuse_near_infrared',
                               'surface_albedo_for_direct_near_infrared',
                               'surface_albedo_for_diffuse_shortwave',
                               'surface_albedo_for_direct_shortwave']:
            state_sw[surface_albedo] = DataArray(
                np.array(float(surface.albedo)),
                attrs={'units': 'dimensionless'})

        for quant in ['shortwave_optical_thickness_due_to_aerosol',
                      'single_scattering_albedo_due_to_aerosol',
                      'aerosol_asymmetry_parameter']:
            state_sw[quant] = DataArray(
                np.zeros((num_sw_bands, numlevels)),
                dims=('num_shortwave_bands', 'mid_levels'),
                attrs={'units': 'dimensionless'})

        return state_lw, state_sw

    def update_cloudy_radiative_state(self, cloud, state0, sw=True):

        # Take cloud quantities from cloud class.
        props = (
           'cloud_ice_particle_size',
           'mass_content_of_cloud_liquid_water_in_atmosphere_layer',
           'mass_content_of_cloud_ice_in_atmosphere_layer',
           'cloud_water_droplet_radius',
           'cloud_area_fraction_in_atmosphere_layer',
        )

        lw_props = (
            'longwave_optical_thickness_due_to_cloud',
        )

        sw_props = (
            'cloud_forward_scattering_fraction',
            'cloud_asymmetry_parameter',
            'shortwave_optical_thickness_due_to_cloud',
            'single_scattering_albedo_due_to_cloud',
        )

        for varname in props + sw_props if sw else props + lw_props:
            state0[varname] = cloud[varname]

    def update_radiative_state(self, atmosphere, surface, state0, sw=True):
        """ Update CliMT formatted atmospheric state using parameters from our
        model.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
            surface (konrad.surface): Surface model.
            state0 (dictionary): atmospheric state in the format for climt
            sw (bool): Toggle between shortwave and longwave calculations.

        Returns:
            dictionary: updated state
        """

        state0['air_temperature'] = DataArray(
            atmosphere['T'][0, :],
            dims=('mid_levels',),
            attrs={'units': 'degK'})

        vmr_h2o = atmosphere['H2O'][0, :]
        specific_humidity = vmr2specific_humidity(vmr_h2o)
        state0['specific_humidity'] = DataArray(
            specific_humidity,
            dims=('mid_levels',),
            attrs={'units': 'g/g'})

        # CliMT/konrad name mapping
        gas_name_mapping = [
            ('mole_fraction_of_methane_in_air', 'CH4'),
            ('mole_fraction_of_carbon_dioxide_in_air', 'CO2'),
            ('mole_fraction_of_nitrous_oxide_in_air', 'N2O'),
            ('mole_fraction_of_ozone_in_air', 'O3'),
            ('mole_fraction_of_cfc11_in_air', 'CFC11'),
            ('mole_fraction_of_cfc12_in_air', 'CFC12'),
            ('mole_fraction_of_cfc22_in_air', 'CFC22'),
            ('mole_fraction_of_carbon_tetrachloride_in_air', 'CCl4'),
            ('mole_fraction_of_oxygen_in_air', 'O2'),
        ]

        for climt_key, konrad_key in gas_name_mapping:
            vmr = atmosphere.get(konrad_key, default=0, keepdims=False)
            state0[climt_key] = DataArray(
                vmr,
                dims=('mid_levels',),
                attrs={'units': 'mole/mole'})

        # Surface quantities
        state0['surface_temperature'] = DataArray(
            np.array(surface['temperature'][-1]),
            attrs={'units': 'degK'})

        if sw:  # properties required only for shortwave
            state0['zenith_angle'] = DataArray(
                np.array(np.deg2rad(self.current_solar_angle)),
                attrs={'units': 'radians'})

        return state0

    def radiative_fluxes(self, atmosphere, surface, cloud):
        """Returns shortwave and longwave fluxes and heating rates.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): atmosphere model
            surface (konrad.surface): surface model
            cloud (konrad.cloud): cloud model

        Returns:
            tuple: containing two dictionaries, one of air temperature
            values and the other of fluxes and heating rates
        """
        if self._state_lw is None or self._state_sw is None:  # first time only
            self._cloud_optical_properties = cloud._rrtmg_cloud_optical_properties
            self._cloud_ice_properties = cloud._rrtmg_cloud_ice_properties
            self._state_lw, self._state_sw = self.init_radiative_state(
                    atmosphere, surface)
            self.update_cloudy_radiative_state(cloud, self._state_lw, sw=False)
            self.update_cloudy_radiative_state(cloud, self._state_sw, sw=True)

        # if there are clouds update the cloud properties for the radiation
        if not isinstance(cloud, ClearSky):
            self.update_cloudy_radiative_state(cloud, self._state_lw, sw=False)
            self.update_cloudy_radiative_state(cloud, self._state_sw, sw=True)

        self.update_radiative_state(atmosphere, surface, self._state_lw,
                                    sw=False)
        self.update_radiative_state(atmosphere, surface, self._state_sw,
                                    sw=True)

        lw_fluxes = self._rad_lw(self._state_lw)
        sw_fluxes = self._rad_sw(self._state_sw)

        return lw_fluxes, sw_fluxes

    def calc_cloudy_nomcica_radiation(self, atmosphere, surface, cloud):

        cloud_fraction = deepcopy(cloud['cloud_area_fraction_in_atmosphere_layer'][:])

        if self._state_sw is None:  # first time only
            cf_cloudy = cloud_fraction[cloud_fraction != 0]
            if cf_cloudy.size > 0:
                if not np.all(cf_cloudy == cf_cloudy[0]):
                    logging.warning(
                        'The konrad implementation of nomcica with partial '
                        'cloud fraction is only suitable for a single '
                        'rectangular cloud. Consider using mcica instead.')

        cf_max = np.max(cloud_fraction)

        # Calculate overcast and clear sky radiative fluxes.
        # Make all the cloudy layers overcast, so that the nomcica version of
        # RRTMG can be used in the shortwave - all wavelengths see cloud.
        # Use the same approach for the longwave for consistency.
        cloud['cloud_area_fraction_in_atmosphere_layer'][cloud_fraction != 0] = 1
        lw_overcast, sw_overcast = self.radiative_fluxes(
            atmosphere, surface, cloud
        )
        lw_fluxes, sw_fluxes = lw_overcast[1], sw_overcast[1]

        # Combine overcast and clear sky fluxes and heating rates to get the
        # all sky fluxes and heating rates.
        for fluxes in lw_fluxes, sw_fluxes:
            for key in fluxes.keys():
                if 'clear_sky' not in key:
                    clear_part = fluxes[key + '_assuming_clear_sky'][:]
                    fluxes[key][:] *= cf_max  # weighted by cloud area fraction
                    fluxes[key][:] += (1 - cf_max) * clear_part

        cloud['cloud_area_fraction_in_atmosphere_layer'][:] *= cloud_fraction

        return lw_fluxes, sw_fluxes

    def calc_radiation(self, atmosphere, surface, cloud):
        """Updates the shortwave, longwave and net heatingrates.
        Converts output from radiative_fluxes to be in the format required for
        our model.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
            surface (konrad.surface): Surface model.
            cloud (konrad.cloud): cloud model
        """
        if not self._is_mcica and not isinstance(cloud, ClearSky):
            if isinstance(cloud, CloudEnsemble):
                weights, cloud_combinations = cloud.get_combinations()

                lw_fluxes = {}
                sw_fluxes = {}
                for weight, cloud_case in zip(weights, cloud_combinations):
                    lw_temp, sw_temp = self.radiative_fluxes(
                        atmosphere, surface, cloud_case)

                    for key, value in lw_temp[1].items():
                        if key not in lw_fluxes.keys():
                            lw_fluxes[key] = weight * value.data
                        else:
                            lw_fluxes[key][:] += weight * value.data

                    for key, value in sw_temp[1].items():
                        if key not in sw_fluxes.keys():
                            sw_fluxes[key] = weight * value.data
                        else:
                            sw_fluxes[key][:] += weight * value.data

            else:
                lw_fluxes, sw_fluxes = self.calc_cloudy_nomcica_radiation(
                    atmosphere, surface, cloud)
        else:
            lw_dT_fluxes, sw_dT_fluxes = self.radiative_fluxes(
                atmosphere, surface, cloud)
            lw_fluxes = lw_dT_fluxes[1]
            sw_fluxes = sw_dT_fluxes[1]

        self['lw_htngrt'] = np.expand_dims(
                lw_fluxes['air_temperature_tendency_from_longwave'].data, 0)
        self['lw_htngrt_clr'] = np.expand_dims(
                lw_fluxes['air_temperature_tendency_from_longwave_assuming_clear_sky'].data, 0)
        self['lw_flxu'] = np.expand_dims(
                lw_fluxes['upwelling_longwave_flux_in_air'].data, 0)
        self['lw_flxd'] = np.expand_dims(
                lw_fluxes['downwelling_longwave_flux_in_air'].data, 0)
        self['lw_flxu_clr'] = np.expand_dims(
                lw_fluxes['upwelling_longwave_flux_in_air_assuming_clear_sky'].data, 0)
        self['lw_flxd_clr'] = np.expand_dims(
                lw_fluxes['downwelling_longwave_flux_in_air_assuming_clear_sky'].data, 0)
        self['sw_htngrt'] = np.expand_dims(
                sw_fluxes['air_temperature_tendency_from_shortwave'].data, 0)
        self['sw_htngrt_clr'] = np.expand_dims(
                sw_fluxes['air_temperature_tendency_from_shortwave_assuming_clear_sky'].data, 0)
        self['sw_flxu'] = np.expand_dims(
                sw_fluxes['upwelling_shortwave_flux_in_air'].data, 0)
        self['sw_flxd'] = np.expand_dims(
                sw_fluxes['downwelling_shortwave_flux_in_air'].data, 0)
        self['sw_flxu_clr'] = np.expand_dims(
                sw_fluxes['upwelling_shortwave_flux_in_air_assuming_clear_sky'].data, 0)
        self['sw_flxd_clr'] = np.expand_dims(
                sw_fluxes['downwelling_shortwave_flux_in_air_assuming_clear_sky'].data, 0)

        self.coords={
            'time': np.array([0]),
            'phlev': atmosphere['phlev'],
            'plev': atmosphere['plev'],
        }
