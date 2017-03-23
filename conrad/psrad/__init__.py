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
            P_sfc, T_sfc, Z, P, T, x_vmr, ...
    """
    p = sounding['P'].values[1:] / 100
    T = sounding['T'].values[1:]
    z = sounding['Z'].values[1:]

    P_sfc = sounding['P'].values[0] / 100
    T_sfc = sounding['T'].values[0]

    ret = [P_sfc, T_sfc, z, p, T]  # P_sfc, T_sfc, Z, P, T

    gases = ['Q', 'O3', 'CO2', 'N2O', 'CO', 'CH4']

    for gas in gases:
        if gas in sounding:
            ret.append(sounding[gas].values[1:] * 1e6)  # Append gas in ppm.
        else:
            ret.append(np.zeros(p.size))

    return tuple(ret)


def psrad_heatingrates(s, albedo=0.05, zenith=41.0, fix_surface=True):
    """Computes the heating rates for a given sounding.

    Parameters:
        s (pd.DataFrame): Atmospheric sounding.
        albedo (float): Surface albedo.
        zenith (float): Zenith angle of the sun.
        fix_surface (bool): Keep the surface temperature constant.
            If `True`, the heating rate for the surface is 0.
            If `False`, use the heating rate of the lowest atmosphere layer.

    Returns:
        pd.DataFrame: Containing pd.Series for the simulated heating rates.
            The keys are 'sw_htngrt', 'lw_htngrt' and 'net_htngrt'.
    """
    dmy_indices = np.asarray([0, 0, 0, 0])
    ic = dmy_indices.astype("int32") + 1
    c_lwc = np.asarray([0., 0., 0., 0.])
    c_iwc = np.asarray([0., 0., 0., 0.])
    c_frc = np.asarray([0., 0., 0., 0.])

    nlev = len(s['P'].values[1:])

    psrad.setup_single(nlev, len(ic), ic, c_lwc, c_iwc, c_frc, zenith, albedo,
                       *_extract_psrad_args(s))

    ret = pd.DataFrame()  # Create pandas DataFrame for return values.

    # Shortwave heating rate.
    psrad.advance_lrtm()  # Perform PSRAD simulation.
    hr = psrad.get_lw_fluxes()[0]  # Ignore additional output.

    # Append a heating rate for the surface (0 or duplicate lowst atmosphere
    # layer).
    hr_sfc = 0 if fix_surface else hr[0, 0]

    # Add surface heating rate and store the results as pands.Series inside
    # the pandas.DataFrame.
    ret['lw_htngrt'] = pd.Series(np.append(hr_sfc, hr), index=s['P'].values)

    # Longwave heating rate. See above for detailed documentation.
    psrad.advance_srtm()
    hr = psrad.get_sw_fluxes()[0]
    hr_sfc = 0 if fix_surface else hr[0, 0]
    ret['sw_htngrt'] = pd.Series(np.append(hr_sfc, hr), index=s['P'].values)

    # Calculate net heating rate.
    ret['net_htngrt'] = ret['sw_htngrt'] + ret['lw_htngrt']

    return ret
