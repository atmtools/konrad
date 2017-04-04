# -*- coding: utf-8 -*-
"""Python wrapper for the PSRAD radiation scheme.
"""
from . import psrad
from xarray import DataArray, Dataset
import numpy as np


__all__ = [
    'psrad_heatingrates',
]


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
            ret.append(np.zeros(p.size))

    return tuple(ret)


def psrad_heatingrates(atmosphere, surface, zenith=41.0):
    """Computes the heating rates for a given atmosphere.

    Parameters:
        atmosphere (pd.DataFrame): Atmospheric atmosphere.
        surface (conrad.surface.Surface): Surface model inherited from abstract
            class `conrad.surface.Surface`.
        zenith (float): Zenith angle of the sun.

    Returns:
        pd.DataFrame: Containing pd.Series for the simulated heating rates.
            The keys are 'sw_htngrt', 'lw_htngrt' and 'net_htngrt'.
    """
    dmy_indices = np.asarray([0, 0, 0, 0])
    ic = dmy_indices.astype("int32") + 1
    c_lwc = np.asarray([0., 0., 0., 0.])
    c_iwc = np.asarray([0., 0., 0., 0.])
    c_frc = np.asarray([0., 0., 0., 0.])

    nlev = atmosphere['plev'].size

    # Extract surface properties.
    P_sfc = surface.pressure / 100
    T_sfc = surface.temperature
    albedo = surface.albedo

    psrad.setup_single(nlev, len(ic), ic, c_lwc, c_iwc, c_frc, zenith,
                       albedo, P_sfc, T_sfc, *_extract_psrad_args(atmosphere))

    ret = Dataset()  # Create xarray.Dataset for return values.

    # Shortwave heating rate.
    psrad.advance_lrtm()  # Perform PSRAD simulation.
    hr = psrad.get_lw_fluxes()[0]  # Ignore additional output.
    ret['lw_htngrt'] = DataArray(hr.ravel(),
                                 coords=[atmosphere['plev']],
                                 dims=['plev'])

    # Longwave heating rate.
    psrad.advance_srtm()
    hr = psrad.get_sw_fluxes()[0]
    ret['sw_htngrt'] = DataArray(hr.ravel(),
                                 coords=[atmosphere['plev']],
                                 dims=['plev'])

    # Net heating rate.
    ret['net_htngrt'] = ret['sw_htngrt'] + ret['lw_htngrt']

    return ret
