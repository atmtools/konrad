"""Define an interface for the RRTMG radiation scheme (through CliMT). """
import numpy as np
import xarray as xr
from sympl import DataArray
from typhon.physics import vmr2specific_humidity

from .radiation import Radiation


__all__ = [
    'RRTMG',
]


class RRTMG(Radiation):
    """RRTMG radiation scheme using the CliMT python wrapper."""

    def __init__(self, *args, solar_constant=510, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_lw = None
        self.state_sw = None

        self.solar_constant = solar_constant

    def update_radiative_state(self, atmosphere, state0, sw=True):
        """ Update CliMT formatted atmospheric state using parameters from our
        model.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
            state0 (dictionary): atmospheric state in the format for climt
            sw (bool): Toggle between shortwave and longwave calculations.

        Returns:
            dictionary: updated state
        """
        plev = atmosphere['plev'].values
        phlev = atmosphere['phlev'].values
        numlevels = len(plev)
        temperature = atmosphere.surface.temperature.data[0]
        albedo = float(atmosphere.surface.albedo.data)
        o2fraction = 0.21
        zenith = np.deg2rad(self.current_solar_angle)

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

        state0['mole_fraction_of_oxygen_in_air'] = DataArray(
            o2fraction * np.ones(numlevels,),
            dims=('mid_levels',),
            attrs={'units': 'mole/mole'})

        # Set trace gas concentrations
        for gasname, gas in [('mole_fraction_of_methane_in_air', 'CH4'),
                             ('mole_fraction_of_carbon_dioxide_in_air', 'CO2'),
                             ('mole_fraction_of_nitrous_oxide_in_air', 'N2O'),
                             ('mole_fraction_of_ozone_in_air', 'O3')]:
            state0[gasname] = DataArray(
                atmosphere[gas][0, :].data,
                dims=('mid_levels',),
                attrs={'units': 'mole/mole'})

        for gasname in ['mole_fraction_of_carbon_tetrachloride_in_air',
                        'mole_fraction_of_cfc22_in_air',
                        'mole_fraction_of_cfc12_in_air',
                        'mole_fraction_of_cfc11_in_air']:
            state0[gasname] = DataArray(
                np.zeros(numlevels,),
                dims=('mid_levels',),
                attrs={'units': 'mole/mole'})

        # Set cloud quantities to zero.
        for quant in ['cloud_ice_particle_size',
                      'mass_content_of_cloud_liquid_water_in_atmosphere_layer',
                      'mass_content_of_cloud_ice_in_atmosphere_layer',
                      'cloud_water_droplet_radius',
                      'cloud_area_fraction_in_atmosphere_layer']:
            units = state0[quant].units
            state0[quant] = DataArray(
                np.zeros(numlevels,),
                dims=('mid_levels',),
                attrs={'units': units})

        if not sw:  # Longwave specific changes
            num_lw_bands = len(state0['num_longwave_bands'])

            state0['longwave_optical_thickness_due_to_cloud'] = DataArray(
                np.zeros((1, num_lw_bands, 1, numlevels)),
                dims=('longitude', 'num_longwave_bands',
                      'latitude', 'mid_levels'),
                attrs={'units': 'dimensionless'})

            state0['longwave_optical_thickness_due_to_aerosol'] = DataArray(
                np.zeros((1, 1, numlevels, num_lw_bands)),
                dims=('longitude', 'latitude',
                      'mid_levels', 'num_longwave_bands'),
                attrs={'units': 'dimensionless'})

        if sw:  # Shortwave specific changes
            num_sw_bands = len(state0['num_shortwave_bands'])

            for quant in ['cloud_forward_scattering_fraction',
                          'cloud_asymmetry_parameter',
                          'shortwave_optical_thickness_due_to_cloud',
                          'single_scattering_albedo_due_to_cloud']:
                state0[quant] = DataArray(
                    np.zeros((1, num_sw_bands, 1, numlevels)),
                    dims=('longitude', 'num_shortwave_bands',
                          'latitude', 'mid_levels'),
                    attrs={'units': 'dimensionless'})

            num_aerosols = len(state0['num_ecmwf_aerosols'])
            state0['aerosol_optical_depth_at_55_micron'] = DataArray(
                np.zeros((1, 1, numlevels, num_aerosols)),
                dims=('longitude', 'latitude',
                      'mid_levels', 'num_ecmwf_aerosols'),
                attrs={'units': 'dimensionless'})

            for quant in ['shortwave_optical_thickness_due_to_aerosol',
                          'single_scattering_albedo_due_to_aerosol',
                          'aerosol_asymmetry_parameter']:
                state0[quant] = DataArray(
                    np.zeros((1, 1, numlevels, num_sw_bands)),
                    dims=('longitude', 'latitude',
                          'mid_levels', 'num_shortwave_bands'),
                    attrs={'units': 'dimensionless'})

        # TODO: Should all the aerosol values be zero?!

        # Surface quantities
        state0['surface_temperature'] = DataArray(
            np.array([[temperature]]),
            dims={'longitude', 'latitude'},
            attrs={'units': 'degK'})

        if sw:  # surface properties required only for shortwave
            for surface_albedo in ['surface_albedo_for_diffuse_near_infrared',
                                   'surface_albedo_for_direct_near_infrared',
                                   'surface_albedo_for_diffuse_shortwave',
                                   'surface_albedo_for_direct_shortwave']:
                state0[surface_albedo] = DataArray(
                    np.array([[albedo]]),
                    dims={'longitude', 'latitude'},
                    attrs={'units': 'dimensionless'})

            # Sun
            state0['zenith_angle'] = DataArray(
                np.array([[zenith]]),
                dims={'longitude', 'latitude'},
                attrs={'units': 'radians'})

        return state0

    def radiative_fluxes(self, atmosphere):
        """Returns shortwave and longwave fluxes and heating rates.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): atmosphere model

        Returns:
            tuple: containing two dictionaries, one of air temperature
            values and the other of fluxes and heating rates
        """
        if self.state_lw is None or self.state_sw is None:
            import climt
            climt.set_constant('stellar_irradiance',
                               value=self.solar_constant,
                               units='W m^-2')
            self.rad_lw = climt.RRTMGLongwave()
            self.rad_sw = climt.RRTMGShortwave(ignore_day_of_year=True)
            self.state_lw = climt.get_default_state([self.rad_lw])
            self.state_sw = climt.get_default_state([self.rad_sw])

        self.update_radiative_state(atmosphere, self.state_lw, sw=False)
        self.update_radiative_state(atmosphere, self.state_sw, sw=True)

        lw_fluxes = self.rad_lw(self.state_lw)
        sw_fluxes = self.rad_sw(self.state_sw)

        return lw_fluxes, sw_fluxes

    def calc_radiation(self, atmosphere):
        """Returns the shortwave, longwave and net heatingrates.
        Converts output from radiative_fluxes to be in the format required for
        our model.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.

        Returns:
            xarray.Dataset: Dataset containing for the simulated heating rates.
                The keys are 'sw_htngrt', 'lw_htngrt' and 'net_htngrt'.
        """
        lw_dT_fluxes, sw_dT_fluxes = self.radiative_fluxes(atmosphere)
        lw_fluxes = lw_dT_fluxes[1]
        sw_fluxes = sw_dT_fluxes[1]

        ret = xr.Dataset({
            # General atmospheric properties.
            'z': atmosphere['z'],

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
                'time': [0],
                'phlev': atmosphere['phlev'].values,
                'plev': atmosphere['plev'].values,
            }
        )

        return ret
