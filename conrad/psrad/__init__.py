# -*- coding: utf-8 -*-
"""Python wrapper for the PSRAD radiation scheme.
"""
from . import psrad
import pandas as pd
import numpy as np


__all__ = [
    'psrad',
    'psrad_lw',
    'psrad_sw',
]


def psrad_lw(s, albedo=0.05, zenith=41.0):
    """Computes the lw radiation for a given sounding.

    Parameters:
        s (pd.DataFrame): Atmospheric sounding.
        albedo (float): Surface albedo.
        zenith (float): Zenith angle of the sun.

    Returns:
        pandas.DataFrame

    .. Author : Bjorn Stevens (bjorn.stevens@mpimet.mpg.de)
    .. Created: 8.10.2016
    """
    dmy_indices = np.asarray([0, 0, 0, 0])
    ic = dmy_indices.astype("int32") + 1
    c_lwc = np.asarray([0., 0., 0., 0.])
    c_iwc = np.asarray([0., 0., 0., 0.])
    c_frc = np.asarray([0., 0., 0., 0.])

    P_sfc = s['P'].values[0] / 100
    T_sfc = s['T'].values[0]

    nlev = len(s['P'].values[1:])

    psrad.setup_single(
        nlev, len(ic), ic, c_lwc, c_iwc, c_frc, zenith, albedo, P_sfc, T_sfc,
        s['Z'].values[1:], s['P'].values[1:] / 100, s['T'].values[1:],
        s['Q'].values[1:] * 1e6, s['O3'].values[1:] * 1e6,
        s['CO2'].values[1:] * 1e6, s['N2O'].values[1:] * 1e6,
        s['CO'].values[1:] * 1e6, s['CH4'].values[1:] * 1e6
        )

    psrad.advance_lrtm()
    hr, hr_clr, lw_flxd, lw_flxd_clr, lw_flxu, lw_flxu_clr = \
        psrad.get_lw_fluxes()
    xhr = np.append([0], hr[0, 0:nlev])
    xhr_clr = np.append([0], hr_clr[0, 0:nlev])

    data = pd.DataFrame({
        'Z': pd.Series(s['Z'].values, index=s['P'].values),
        'P': pd.Series(s['P'].values, index=s['P'].values),
        'T': pd.Series(s['T'].values, index=s['P'].values),
        'Q': pd.Series(s['Q'].values, index=s['P'].values),
        'lw_htngrt': pd.Series(xhr, index=s['P'].values),
        'lw_htngrt_clr': pd.Series(xhr_clr, index=s['P'].values),
        'lw_flxu': pd.Series(lw_flxu[0, 0:nlev + 1], index=s['P'].values),
        'lw_flxd': pd.Series(lw_flxd[0, 0:nlev + 1], index=s['P'].values),
        'lw_flxu_clr': pd.Series(lw_flxu_clr[0, 0:nlev + 1],
                                 index=s['P'].values),
        'lw_flxd_clr': pd.Series(lw_flxd_clr[0, 0:nlev + 1],
                                 index=s['P'].values)
        })

    return data


def psrad_sw(s, albedo=0.05, zenith=41.0):
    """Computes the sw radiation for a given sounding.

    Parameters:
        s (pd.DataFrame): Atmospheric sounding.
        albedo (float): Surface albedo.
        zenith (float): Zenith angle of the sun.

    .. Author: Theresa Lang
    .. Created: 19.12.16
    """
    dmy_indices = np.asarray([0, 0, 0, 0])
    ic = dmy_indices.astype("int32") + 1
    c_lwc = np.asarray([0., 0., 0., 0.])
    c_iwc = np.asarray([0., 0., 0., 0.])
    c_frc = np.asarray([0., 0., 0., 0.])

    P_sfc = s['P'].values[0] / 100
    T_sfc = s['T'].values[0]

    nlev = len(s['P'].values[1:])

    psrad.setup_single(
        nlev, len(ic), ic, c_lwc, c_iwc, c_frc, zenith, albedo, P_sfc, T_sfc,
        s['Z'].values[1:], s['P'].values[1:] / 100, s['T'].values[1:],
        s['Q'].values[1:] * 1e6, s['O3'].values[1:] * 1e6,
        s['CO2'].values[1:] * 1e6, s['N2O'].values[1:] * 1e6,
        s['CO'].values[1:] * 1e6, s['CH4'].values[1:] * 1e6
        )

    psrad.advance_srtm()
    hr, hr_clr, sw_flxd, sw_flxd_clr, sw_flxu, sw_flxu_clr, vis_frc, \
        par_dn, nir_dff, vis_diff, par_diff = \
        psrad.get_sw_fluxes()
    xhr = np.append([0], hr[0, 0:nlev])
    xhr_clr = np.append([0], hr_clr[0, 0:nlev])

    data = pd.DataFrame({
        'Z': pd.Series(s['Z'].values, index=s['P'].values),
        'P': pd.Series(s['P'].values, index=s['P'].values),
        'T': pd.Series(s['T'].values, index=s['P'].values),
        'Q': pd.Series(s['Q'].values, index=s['P'].values),
        'sw_htngrt': pd.Series(xhr, index=s['P'].values),
        'sw_htngrt_clr': pd.Series(xhr_clr, index=s['P'].values),
        'sw_flxu': pd.Series(sw_flxu[0, 0:nlev + 1], index=s['P'].values),
        'sw_flxd': pd.Series(sw_flxd[0, 0:nlev + 1], index=s['P'].values),
        'sw_flxu_clr': pd.Series(sw_flxu_clr[0, 0:nlev + 1],
                                 index=s['P'].values),
        'sw_flxd_clr': pd.Series(sw_flxd_clr[0, 0:nlev + 1],
                                 index=s['P'].values)
        })

    return data
