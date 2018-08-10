"""Define an interface for the RRTMG radiation scheme (through CliMT). """
import numpy as np
import xarray as xr
from sympl import DataArray
from typhon.physics import vmr2specific_humidity

from .radiation import Radiation
from konrad.cloud import ClearSky

__all__ = [
    'RRTMG',
]


class RRTMG(Radiation):
    """RRTMG radiation scheme using the CliMT python wrapper."""

    def __init__(self, *args, solar_constant=510, **kwargs):
        super().__init__(*args, **kwargs)
        self._state_lw = None
        self._state_sw = None

        self._rad_lw = None
        self._rad_sw = None

        self.solar_constant = solar_constant

    def init_radiative_state(self, atmosphere):

        import climt
        climt.set_constant('stellar_irradiance',
                           value=self.solar_constant,
                           units='W m^-2')
        self._rad_lw = climt.RRTMGLongwave()
        self._rad_sw = climt.RRTMGShortwave(ignore_day_of_year=True)
        state_lw = climt.get_default_state([self._rad_lw])
        state_sw = climt.get_default_state([self._rad_sw])

        plev = atmosphere['plev'].values
        phlev = atmosphere['phlev'].values
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

        ### Aerosols ###
        # TODO: Should all the aerosol values be zero?!
        # Longwave specific
        num_lw_bands = len(state_lw['num_longwave_bands'])

        state_lw['longwave_optical_thickness_due_to_aerosol'] = DataArray(
                np.zeros((1, 1, numlevels, num_lw_bands)),
                dims=('longitude', 'latitude',
                      'mid_levels', 'num_longwave_bands'),
                attrs={'units': 'dimensionless'})

        # Shortwave specific changes
        num_sw_bands = len(state_sw['num_shortwave_bands'])

        num_aerosols = len(state_sw['num_ecmwf_aerosols'])
        state_sw['aerosol_optical_depth_at_55_micron'] = DataArray(
                np.zeros((1, 1, numlevels, num_aerosols)),
                dims=('longitude', 'latitude',
                      'mid_levels', 'num_ecmwf_aerosols'),
                attrs={'units': 'dimensionless'})

        for quant in ['shortwave_optical_thickness_due_to_aerosol',
                      'single_scattering_albedo_due_to_aerosol',
                      'aerosol_asymmetry_parameter']:
            state_sw[quant] = DataArray(
                np.zeros((1, 1, numlevels, num_sw_bands)),
                dims=('longitude', 'latitude',
                      'mid_levels', 'num_shortwave_bands'),
                attrs={'units': 'dimensionless'})

        return state_lw, state_sw

    def update_cloudy_radiative_state(self, cloud, state0, sw=True):

        # Take cloud quantities from cloud class.
        state0['cloud_ice_particle_size'] = cloud.cloud_ice_particle_size
        state0['mass_content_of_cloud_liquid_water_in_atmosphere_layer'] = cloud.mass_content_of_cloud_liquid_water_in_atmosphere_layer
        state0['mass_content_of_cloud_ice_in_atmosphere_layer'] = cloud.mass_content_of_cloud_ice_in_atmosphere_layer
        state0['cloud_water_droplet_radius'] = cloud.cloud_water_droplet_radius
        state0['cloud_area_fraction_in_atmosphere_layer'] = cloud.cloud_area_fraction_in_atmosphere_layer

        if not sw:
            state0['longwave_optical_thickness_due_to_cloud'] = cloud.longwave_optical_thickness_due_to_cloud
        else:
            state0['cloud_forward_scattering_fraction'] = cloud.cloud_forward_scattering_fraction
            state0['cloud_asymmetry_parameter'] = cloud.cloud_asymmetry_parameter
            state0['shortwave_optical_thickness_due_to_cloud'] = cloud.shortwave_optical_thickness_due_to_cloud
            state0['single_scattering_albedo_due_to_cloud'] = cloud.single_scattering_albedo_due_to_cloud

        return

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
            atmosphere['T'][0, :].data,
            dims=('mid_levels',),
            attrs={'units': 'degK'})

        vmr_h2o = atmosphere['H2O'][0, :].data
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
            state0[climt_key] = DataArray(
                atmosphere.get_values(konrad_key, default=0, keepdims=False),
                dims=('mid_levels',),
                attrs={'units': 'mole/mole'})

        # Surface quantities
        state0['surface_temperature'] = DataArray(
<<<<<<< HEAD
            np.array([[surface.temperature.data[0]]]),
=======
            np.array([[surface['temperature'][-1]]]),
>>>>>>> a8fd0b1... Fix radiation
            dims={'longitude', 'latitude'},
            attrs={'units': 'degK'})

        if sw:  # surface properties required only for shortwave
            for surface_albedo in ['surface_albedo_for_diffuse_near_infrared',
                                   'surface_albedo_for_direct_near_infrared',
                                   'surface_albedo_for_diffuse_shortwave',
                                   'surface_albedo_for_direct_shortwave']:
                state0[surface_albedo] = DataArray(
                    np.array([[float(surface.albedo.data)]]),
                    dims={'longitude', 'latitude'},
                    attrs={'units': 'dimensionless'})

            # Sun
            state0['zenith_angle'] = DataArray(
                np.array([[np.deg2rad(self.current_solar_angle)]]),
                dims={'longitude', 'latitude'},
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
        if self._state_lw is None or self._state_sw is None: # first time only
            self._state_lw, self._state_sw = self.init_radiative_state(atmosphere)
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

    def calc_radiation(self, atmosphere, surface, cloud):
        """Returns the shortwave, longwave and net heatingrates.
        Converts output from radiative_fluxes to be in the format required for
        our model.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
            surface (konrad.surface): Surface model.
            cloud (konrad.cloud): cloud model

        Returns:
            xarray.Dataset: Dataset containing for the simulated heating rates.
                The keys are 'sw_htngrt', 'lw_htngrt' and 'net_htngrt'.
        """
        lw_dT_fluxes, sw_dT_fluxes = self.radiative_fluxes(atmosphere, surface,
                                                           cloud)
        lw_fluxes = lw_dT_fluxes[1]
        sw_fluxes = sw_dT_fluxes[1]

        ret = xr.Dataset({
            # Longwave fluxes and heatingrates.
            'lw_htngrt': (
                ['time', 'plev'],
                lw_fluxes['longwave_heating_rate'][0].data),
            'lw_htngrt_clr': (
                ['time', 'plev'],
                lw_fluxes['longwave_heating_rate_assuming_clear_sky'][0].data),
            'lw_flxu': (
                ['time', 'phlev'],
                lw_fluxes['upwelling_longwave_flux_in_air'][0].data),
            'lw_flxd': (
                ['time', 'phlev'],
                lw_fluxes['downwelling_longwave_flux_in_air'][0].data),
            'lw_flxu_clr': (
                ['time', 'phlev'],
                lw_fluxes['upwelling_longwave_flux_in_air_assuming_clear_sky'][0].data),
            'lw_flxd_clr': (
                ['time', 'phlev'],
                lw_fluxes['downwelling_longwave_flux_in_air_assuming_clear_sky'][0].data),

            # Shortwave fluxes and heatingrates.
            'sw_htngrt': (
                ['time', 'plev'],
                sw_fluxes['shortwave_heating_rate'][0].data),
            'sw_htngrt_clr': (
                ['time', 'plev'],
                sw_fluxes['shortwave_heating_rate_assuming_clear_sky'][0].data),
            'sw_flxu': (
                ['time', 'phlev'],
                sw_fluxes['upwelling_shortwave_flux_in_air'][0].data),
            'sw_flxd': (
                ['time', 'phlev'],
                sw_fluxes['downwelling_shortwave_flux_in_air'][0].data),
            'sw_flxu_clr': (
                ['time', 'phlev'],
                sw_fluxes['upwelling_shortwave_flux_in_air_assuming_clear_sky'][0].data),
            'sw_flxd_clr': (
                ['time', 'phlev'],
                sw_fluxes['downwelling_shortwave_flux_in_air_assuming_clear_sky'][0].data),
        },
            coords={
                'time': np.array([0]),
                'phlev': atmosphere['phlev'],
                'plev': atmosphere['plev'],
            }
        )

        return ret
