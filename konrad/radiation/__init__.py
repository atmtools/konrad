# -*- coding: utf-8 -*-
"""Module containing classes describing different radiation models.
"""
import abc
import logging

import numpy as np
import xarray as xr
from typhon.physics import vmr2specific_humidity

from konrad import constants
from konrad.utils import append_description


logger = logging.getLogger()

__all__ = [
    'Radiation',
    'PSRAD',
    'RRTMG',
]

# Subclasses of `Radiation` need to define a `Radiation.calc_radiation()`
# method that returns a `xr.Dataset` containing at least following variables:
REQUIRED_VARIABLES = [
    'net_htngrt',
    'lw_flxd',
    'lw_flxu',
    'sw_flxd',
    'sw_flxu',
]


class Radiation(metaclass=abc.ABCMeta):
    """Abstract base class to define requirements for radiation models."""
    def __init__(self, zenith_angle=47.88, diurnal_cycle=False, bias=None):
        """Return a radiation model.

        Parameters:
            zenith_angle (float): Zenith angle of the sun.
                The default angle of 47.88 degree results in 342 W/m^2
                solar insolation at the top of the atmosphere when used
                together with a solar constant of 510 W/m^2.
            diurnal_cycle (bool): Toggle diurnal cycle of solar angle.
            bias (dict-like): A dict-like object that stores bias
                corrections for the diagnostic variable specified by its key,
                e.g. `bias = {'net_htngrt': 2}`.
        """
        super().__init__()
        
        self.zenith_angle = zenith_angle
        self.diurnal_cycle = diurnal_cycle

        self.current_solar_angle = 0

        self.bias = bias

    @abc.abstractmethod
    def calc_radiation(self, atmosphere):
        return xr.Dataset()

    def get_heatingrates(self, atmosphere):
        """Returns `xr.Dataset` containing radiative transfer results."""
        rad_dataset = self.calc_radiation(atmosphere)

        self.correct_bias(rad_dataset)

        self.derive_diagnostics(rad_dataset)

        append_description(rad_dataset)

        self.check_dataset(rad_dataset)

        return rad_dataset

    @staticmethod
    def check_dataset(dataset):
        """Check if a given dataset contains all required variables."""
        for key in REQUIRED_VARIABLES:
            if key not in dataset.variables:
                raise KeyError(
                    f'"{key}" not present in radiative transfer results.'
                )

    def correct_bias(self, dataset):
        """Apply bias correction."""
        if self.bias is not None:
            for key, value in self.bias.items():
                if key not in dataset.indexes:
                    dataset[key] -= value

        # TODO: Add interpolation for biases passed as `xr.Dataset`.

    def derive_diagnostics(self, dataset):
        """Derive diagnostic variables from radiative transfer results."""
        # Net heating rate.
        dataset['net_htngrt'] = xr.DataArray(
            data=dataset.lw_htngrt + dataset.sw_htngrt,
            dims=['time', 'plev'],
        )

        # Radiation budget at top of the atmosphere (TOA).
        dataset['toa'] = xr.DataArray(
            data=((dataset.sw_flxd.values[:, -1] +
                   dataset.lw_flxd.values[:, -1]) -
                  (dataset.sw_flxu.values[:, -1] +
                   dataset.lw_flxu.values[:, -1])
                  ),
            dims=['time'],
        )

    @staticmethod
    def heatingrates_from_fluxes(pressure, downward_flux, upward_flux):
        """Calculate heating rates from radiative fluxes.

        Parameters:
            pressure (ndarray): Pressure half-levels [Pa].
            downward_flux (ndarray): Downward radiative flux [W/m^2].
            upward_flux (ndarray): Upward radiative flux [W/m^2].

        Returns:
            ndarray: Radiative heating rate [K/day].
        """
        c_p = constants.isobaric_mass_heat_capacity
        g = constants.earth_standard_gravity

        Q = g / c_p *  np.diff(upward_flux - downward_flux) / np.diff(pressure)
        Q *= 3600 * 24

        return Q

    def adjust_solar_angle(self, time):
        """Adjust the zenith angle of the sun according to time of day.

        Parameters:
            time (float): Current time [days].
        """
        # When the diurnal cycle is disabled, use the constant zenith angle.
        if not self.diurnal_cycle:
            self.current_solar_angle = self.zenith_angle
            return

        # The local zenith angle, calculated from the latitude and longitude.
        # Seasons are not considered.
        solar_angle = 180/np.pi * np.arccos(
                np.cos(np.deg2rad(self.zenith_angle)
                ) * np.cos(2 * np.pi * time))
        
        self.current_solar_angle = solar_angle


class PSRAD(Radiation):
    """Radiation model using the ICON PSRAD radiation scheme.

    The PSRAD class relies on external radiative-transfer code. Currently this
    functionality is provided by the PSRAD radiation scheme. You need a compiled
    version of this code in order to install and run ``konrad``.

    A stable version is accessible through the internal subversion repository:

        $ svn co https://arts.mi.uni-hamburg.de/svn/internal/browser/psrad/trunk psrad

    Follow the instructions given in the repository to compile PSRAD on your
    machine. A part of the installation process is to set some environment
    variables. Thos are also needed in order to run ``konrad``:

        $ source config/psrad_env.bash

    """
    @staticmethod
    def _extract_psrad_args(atmosphere):
        """Returns tuple of mixing ratios to use with psrad.

        Paramteres:
            atmosphere (dict or pandas.DataFrame): Atmospheric atmosphere.

        Returns:
            tuple(ndarray): ndarrays in the order and unit to use with `psrad`:
                Z, P, T, x_vmr, ...
        """
        z = atmosphere['z'].values
        p = atmosphere['plev'].values / 100
        T = atmosphere['T'].values

        ret = [z, p, T]  # Z, P, T

        # Keep order as it is expected by PSRAD.
        required_gases = ['H2O', 'O3', 'CO2', 'N2O', 'CO', 'CH4']

        for gas in required_gases:
            if gas in atmosphere:
                ret.append(atmosphere[gas].values * 1e6)  # Append gas in ppm.
            else:
                ret.append(np.zeros(np.size(p)))

        return tuple(ret)

    def calc_radiation(self, atmosphere):
        """Returns the shortwave, longwave and net heatingrates.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.

        Returns:
            xarray.Dataset: Dataset containing for the simulated heating rates.
                The keys are 'sw_htngrt', 'lw_htngrt' and 'net_htngrt'.
        """
        from . import psrad
        self.psrad = psrad

        dmy_indices = np.asarray([0, 0, 0, 0])
        ic = dmy_indices.astype("int32") + 1
        c_lwc = np.asarray([0., 0., 0., 0.])
        c_iwc = np.asarray([0., 0., 0., 0.])
        c_frc = np.asarray([0., 0., 0., 0.])

        nlev = atmosphere['plev'].size

        # Extract surface properties.
        P_sfc = atmosphere.surface.pressure.values / 100
        T_sfc = atmosphere.surface.temperature.values[0]
        albedo = atmosphere.surface.albedo.values

        # Use the **current** solar angle as zenith angle for the simulation.
        zenith = self.current_solar_angle

        self.psrad.setup_single(nlev, len(ic), ic, c_lwc, c_iwc, c_frc, zenith,
                                albedo, P_sfc, T_sfc,
                                *self._extract_psrad_args(atmosphere))

        self.psrad.advance_lrtm()  # Longwave simulations.
        (lw_hr, lw_hr_clr, lw_flxd,
         lw_flxd_clr, lw_flxu, lw_flxu_clr) = self.psrad.get_lw_fluxes()

        self.psrad.advance_srtm()  # Shortwave simulations.

        (sw_hr, sw_hr_clr, sw_flxd,
         sw_flxd_clr, sw_flxu, sw_flxu_clr,
         vis_frc, par_dn, nir_dff, vis_diff,
         par_diff) = self.psrad.get_sw_fluxes()

        ret = xr.Dataset({
            # General atmospheric properties.
            'z': atmosphere['z'],
            # Longwave fluxes and heatingrates.
            'lw_htngrt': (['time', 'plev'], lw_hr[:, :]),
            'lw_htngrt_clr': (['time', 'plev'], lw_hr_clr[:, :]),
            'lw_flxu': (['time', 'phlev'], lw_flxu[:, :]),
            'lw_flxd': (['time', 'phlev'], lw_flxd[:, :]),
            'lw_flxu_clr': (['time', 'phlev'], lw_flxu_clr[:, :]),
            'lw_flxd_clr': (['time', 'phlev'], lw_flxd_clr[:, :]),
            # Shortwave fluxes and heatingrates.
            # Note: The shortwave fluxes and heatingrates calculated by PSRAD
            # are **inverted**. Therefore, they are flipped to make the input
            # and output of this function consistent.
            'sw_htngrt': (['time', 'plev'], sw_hr[:, ::-1]),
            'sw_htngrt_clr': (['time', 'plev'], sw_hr_clr[:, ::-1]),
            'sw_flxu': (['time', 'phlev'], sw_flxu[:, ::-1]),
            'sw_flxd': (['time', 'phlev'], sw_flxd[:, ::-1]),
            'sw_flxu_clr': (['time', 'phlev'], sw_flxu_clr[:, ::-1]),
            'sw_flxd_clr': (['time', 'phlev'], sw_flxd_clr[:, ::-1]),
            },
            coords={
                'time': [0],
                'plev': atmosphere['plev'].values,
                'phlev': atmosphere['phlev'].values,
            }
            )

        return ret


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
        
        Returns:
            dictionary: updated state
        """
        from sympl import DataArray 

        plev = atmosphere['plev'].values
        phlev = atmosphere['phlev'].values
        numlevels = len(plev)
        temperature = atmosphere.surface.temperature.data[0]
        albedo = float(atmosphere.surface.albedo.data)
        o2fraction = 0.21
        zenith = np.deg2rad(self.current_solar_angle)

        state0['mid_levels'] = DataArray(np.arange(0, numlevels),
              dims=('mid_levels'), attrs={'label': 'mid_levels'})
        state0['interface_levels'] = DataArray(np.arange(0, numlevels+1),
              dims=('interface_levels'), attrs={'label': 'interface_levels'})
        state0['air_pressure'] = DataArray(plev,
              dims=('mid_levels'), attrs={'units': 'Pa'})
        state0['air_pressure_on_interface_levels'] = DataArray(phlev,
              dims=('interface_levels'), attrs={'units': 'Pa'})
        state0['air_temperature'] = DataArray(atmosphere['T'][0, :].data,
              dims=('mid_levels'), attrs={'units': 'degK'})
        vmr_h2o = atmosphere['H2O'][0, :].data
        specific_humidity = vmr2specific_humidity(vmr_h2o)
        state0['specific_humidity'] = DataArray(specific_humidity,
              dims=('mid_levels'), attrs={'units': 'g/g'})
        state0['mole_fraction_of_oxygen_in_air'] = DataArray(
                o2fraction*np.ones(numlevels,), dims=('mid_levels'),
                attrs={'units': 'mole/mole'})
        # Set trace gas concentrations
        for gasname, gas in [('mole_fraction_of_methane_in_air', 'CH4'),
                             ('mole_fraction_of_carbon_dioxide_in_air', 'CO2'),
                             ('mole_fraction_of_nitrous_oxide_in_air', 'N2O'),
                             ('mole_fraction_of_ozone_in_air', 'O3')]:
            state0[gasname] = DataArray(atmosphere[gas][0, :].data,
                  dims=('mid_levels'), attrs={'units': 'mole/mole'})
        for gasname in ['mole_fraction_of_carbon_tetrachloride_in_air',
                        'mole_fraction_of_cfc22_in_air',
                        'mole_fraction_of_cfc12_in_air',
                        'mole_fraction_of_cfc11_in_air']:
            state0[gasname] = DataArray(np.zeros(numlevels,), dims=('mid_levels'),
                  attrs={'units': 'mole/mole'})
        
        # Set cloud quantities to zero.
        for quant in ['cloud_ice_particle_size',
                      'mass_content_of_cloud_liquid_water_in_atmosphere_layer',
                      'mass_content_of_cloud_ice_in_atmosphere_layer',
                      'cloud_water_droplet_radius',
                      'cloud_area_fraction_in_atmosphere_layer']:
            units = state0[quant].units
            state0[quant] = DataArray(np.zeros(numlevels,), dims=('mid_levels'),
                  attrs={'units': units})
        
        if not sw: # Longwave specific changes
            num_lw_bands = len(state0['num_longwave_bands'])
            state0['longwave_optical_thickness_due_to_cloud'] = DataArray(
                    np.zeros((1, num_lw_bands, 1, numlevels)),
                    dims=('longitude', 'num_longwave_bands', 'latitude', 'mid_levels'),
                    attrs={'units': 'dimensionless'})
            state0['longwave_optical_thickness_due_to_aerosol'] = DataArray(
                    np.zeros((1, 1, numlevels, num_lw_bands)),
                    dims=('longitude', 'latitude', 'mid_levels', 'num_longwave_bands'),
                    attrs={'units': 'dimensionless'})
        if sw: # Shortwave specific changes
            num_sw_bands = len(state0['num_shortwave_bands'])
            for quant in ['cloud_forward_scattering_fraction', 'cloud_asymmetry_parameter',
                          'shortwave_optical_thickness_due_to_cloud',
                          'single_scattering_albedo_due_to_cloud']:
                state0[quant] = DataArray(np.zeros((1, num_sw_bands, 1, numlevels)),
                         dims=('longitude', 'num_shortwave_bands', 'latitude',
                               'mid_levels'),
                         attrs={'units': 'dimensionless'})
            num_aerosols = len(state0['num_ecmwf_aerosols'])
            state0['aerosol_optical_depth_at_55_micron'] = DataArray(
                    np.zeros((1, 1, numlevels, num_aerosols)),
                    dims=('longitude', 'latitude', 'mid_levels', 'num_ecmwf_aerosols'),
                    attrs={'units': 'dimensionless'})
            for quant in ['shortwave_optical_thickness_due_to_aerosol',
                          'single_scattering_albedo_due_to_aerosol',
                          'aerosol_asymmetry_parameter']:
                state0[quant] = DataArray(np.zeros((1, 1, numlevels, num_sw_bands)),
                         dims=('longitude', 'latitude', 'mid_levels', 'num_shortwave_bands'),
                         attrs={'units': 'dimensionless'})
        #TODO: should all the aerosol values be zero?!
        
        # Surface quantities
        state0['surface_temperature'] = DataArray(np.array([[temperature]]),
              dims={'longitude', 'latitude'},
              attrs={'units': 'degK'})
        if sw: # surface properties required only for shortwave
            for surface_albedo in ['surface_albedo_for_diffuse_near_infrared',
                           'surface_albedo_for_direct_near_infrared',
                           'surface_albedo_for_diffuse_shortwave',
                           'surface_albedo_for_direct_shortwave']:
                state0[surface_albedo] = DataArray(np.array([[albedo]]), 
                         dims={'longitude', 'latitude'}, attrs={'units': 'dimensionless'})
            # Sun
            state0['zenith_angle'] = DataArray(np.array([[zenith]]),
                     dims={'longitude', 'latitude'}, attrs={'units': 'radians'})
        
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
            'lw_htngrt': (['time', 'plev'],
                          lw_fluxes['longwave_heating_rate'][0].data),
            'lw_htngrt_clr': (['time', 'plev'],
                              lw_fluxes['longwave_heating_rate_assuming_clear_sky'][0].data),
            'lw_flxu': (['time', 'phlev'],
                        lw_fluxes['upwelling_longwave_flux_in_air'][0].data),
            'lw_flxd': (['time', 'phlev'],
                        lw_fluxes['downwelling_longwave_flux_in_air'][0].data),
            'lw_flxu_clr': (['time', 'phlev'],
                            lw_fluxes['upwelling_longwave_flux_in_air_assuming_clear_sky'][0].data),
            'lw_flxd_clr': (['time', 'phlev'],
                            lw_fluxes['downwelling_longwave_flux_in_air_assuming_clear_sky'][0].data),
            # Shortwave fluxes and heatingrates.
            'sw_htngrt': (['time', 'plev'],
                          sw_fluxes['shortwave_heating_rate'][0].data),
            'sw_htngrt_clr': (['time', 'plev'],
                              sw_fluxes['shortwave_heating_rate_assuming_clear_sky'][0].data),
            'sw_flxu': (['time', 'phlev'], sw_fluxes['upwelling_shortwave_flux_in_air'][0].data),
            'sw_flxd': (['time', 'phlev'], sw_fluxes['downwelling_shortwave_flux_in_air'][0].data),
            'sw_flxu_clr': (['time', 'phlev'],
                            sw_fluxes['upwelling_shortwave_flux_in_air_assuming_clear_sky'][0].data),
            'sw_flxd_clr': (['time', 'phlev'],
                            sw_fluxes['downwelling_shortwave_flux_in_air_assuming_clear_sky'][0].data),
            },
            coords={
                'time': [0],
                'phlev': atmosphere['phlev'].values,
                'plev': atmosphere['plev'].values,
            }
            )

        return ret
