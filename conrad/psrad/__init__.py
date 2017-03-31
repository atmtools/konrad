# -*- coding: utf-8 -*-
"""Python wrapper for the PSRAD radiation scheme.
"""
from . import psrad
import pandas as pd
import numpy as np


__all__ = [
    'psrad_heatingrates',
]


def _extract_psrad_args(sounding):
    """Returns tuple of mixing ratios to use with psrad.

    Paramteres:
        sounding (dict or pandas.DataFrame): Atmospheric sounding.

    Returns:
        tuple(ndarray): ndarrays in the order and unit to use with `psrad`:
            Z, P, T, x_vmr, ...
    """
    p = sounding['P'].values / 100
    T = sounding['T'].values
    z = sounding['Z'].values

    ret = [z, p, T]  # Z, P, T

    required_gases = ['Q', 'O3', 'CO2', 'N2O', 'CO', 'CH4']

    for gas in required_gases:
        if gas in sounding:
            ret.append(sounding[gas].values * 1e6)  # Append gas in ppm.
        else:
            ret.append(np.zeros(p.size))

    return tuple(ret)


def psrad_heatingrates(sounding, surface, zenith=41.0):
    """Computes the heating rates for a given sounding.

    Parameters:
        sounding (pd.DataFrame): Atmospheric sounding.
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

    nlev = len(sounding['P'].values)

    # Extract surface properties.
    P_sfc = surface.pressure / 100
    T_sfc = surface.temperature
    albedo = surface.albedo

    psrad.setup_single(nlev, len(ic), ic, c_lwc, c_iwc, c_frc, zenith,
                       albedo, P_sfc, T_sfc, *_extract_psrad_args(sounding))

    ret = pd.DataFrame()  # Create pandas DataFrame for return values.

    # Shortwave heating rate.
    psrad.advance_lrtm()  # Perform PSRAD simulation.
    hr = psrad.get_lw_fluxes()[0]  # Ignore additional output.
    ret['lw_htngrt'] = pd.Series(hr.ravel(), index=sounding['P'].values)

    # Longwave heating rate.
    psrad.advance_srtm()
    hr = psrad.get_sw_fluxes()[0]
    ret['sw_htngrt'] = pd.Series(hr.ravel(), index=sounding['P'].values)

    # Net heating rate.
    ret['net_htngrt'] = ret['sw_htngrt'] + ret['lw_htngrt']

    return ret
