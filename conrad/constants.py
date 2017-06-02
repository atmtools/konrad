# -*- coding: utf-8 -*-
"""Physical constants.
"""
import scipy.constants as spc


# Phyiscal constants
Cp = isobaric_mass_heat_capacity = 1003.5  # J kg^-1 K^-1
g = earth_standard_gravity = spc.g  # m s^-2
stefan_boltzmann = 5.67e-8  # W m^-2 K^-4

# Variable descriptions
variable_description = {
    'plev': {
        'units': 'Pa',
        'standard_name': 'air_pressure',
        },
    'phlev': {
        'units': 'Pa',
        'standard_name': 'air_pressure_at_halflevel',
    },
    'time': {
        'standard_name': 'time',
        'units': 'hours since 0001-01-01 00:00:00.0',
        'calender': 'gregorian',
        },
    'T': {
        'units': 'K',
        'standard_name': 'air_temperature',
        'arts_name': 'T',
        },
    'z': {
        'units': 'm',
        'standard_name': 'height',
        'arts_name': 'z',
        },
    'H2O': {
        'units': '1',
        'standard_name': 'humidity_mixing_ratio',
        'arts_name': 'abs_species-H2O',
        },
    'N2O': {
        'units': '1',
        'standard_name': 'nitrogene_mixing_ratio',
        'arts_name': 'abs_species-N2O',
        },
    'O3': {
        'units': '1',
        'standard_name': 'ozone_mixing_ratio',
        'arts_name': 'abs_species-O3',
        },
    'CO2': {
        'units': '1',
        'standard_name': 'carbon_dioxide_mixing_ratio',
        'arts_name': 'abs_species-CO2',
        },
    'CO': {
        'units': '1',
        'standard_name': 'carbon_monoxide_mixing_ratio',
        'arts_name': 'abs_species-CO',
        },
    'CH4': {
        'units': '1',
        'standard_name': 'methane_mixing_ratio',
        'arts_name': 'abs_species-CH4',
        },
    'lw_htngrt': {
        'units': 'K / day',
        'standard_name': 'tendency_of_air_temperature_due_to_longwave_heating',
        },
    'lw_htngrt_clr': {
        'units': 'K / day',
        'standard_name': ('tendency_of_air_temperature_'
                          'due_to_longwave_heating_assuming_clear_sky'
                          ),
        },
    'lw_flxu': {
        'units': 'W / m**2',
        'standard_name': 'upwelling_longwave_flux_in_air',
        },
    'lw_flxd': {
        'units': 'W / m**2',
        'standard_name': 'downwelling_longwave_flux_in_air',
        },
    'lw_flxu_clr': {
        'units': 'W / m**2',
        'standard_name': 'upwelling_longwave_flux_in_air_assuming_clear_sky',
        },
    'lw_flxd_clr': {
        'units': 'W / m**2',
        'standard_name': 'downwelling_longwave_flux_in_air_assuming_clear_sky',
        },
    'sw_htngrt': {
        'units': 'K / day',
        'standard_name':
            'tendency_of_air_temperature_due_to_shortwave_heating',
        },
    'sw_htngrt_clr': {
        'units': 'K / day',
        'standard_name': ('tendency_of_air_temperature_'
                          'due_to_shortwave_heating_assuming_clear_sky'
                          ),
        },
    'sw_flxu': {
        'units': 'W / m**2',
        'standard_name': 'upwelling_shortwave_flux_in_air',
        },
    'sw_flxd': {
        'units': 'W / m**2',
        'standard_name': 'downwelling_shortwave_flux_in_air',
        },
    'sw_flxu_clr': {
        'units': 'W / m**2',
        'standard_name': 'upwelling_shortwave_flux_in_air_assuming_clear_sky',
        },
    'sw_flxd_clr': {
        'units': 'W / m**2',
        'standard_name':
            'downwelling_shortwave_flux_in_air_assuming_clear_sky',
        },
    'net_htngrt': {
        'units': 'K / day',
        'standard_name':
            'tendency_of_air_temperature_due_to_radiative_heating',
        },
    'deltaT': {
        'units': 'K / day',
        'standard_name': 'tendency_of_air_temperature',
        },
}
