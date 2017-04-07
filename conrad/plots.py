# -*- coding: utf-8 -*-
"""Plotting related functions.
"""
from matplotlib.ticker import FuncFormatter
import numpy as np
import typhon.plots


__all__ = [
    'plot_overview_p_log',
    'plot_overview_z',
]


@FuncFormatter
def _percent_formatter(x, pos):
    return '{:.0f}\N{SIX-PER-EM SPACE}%'.format(x * 100)


def plot_overview_p_log(data, lw_htngrt, sw_htngrt, axes, **kwargs):
    """Plot overview of atmopsheric temperature and humidity profiles.

    Parameters:
        data:
        lw_htngrt:
        sw_htngrt:
        axes (list, tuple or ndarray): Three AxesSubplots.
        **kwargs: Additional keyword arguments passed to all calls
            of `atmospheric_profile`.
    """
    if len(axes) != 3:
        raise Exception('Need to pass three AxesSubplot.')
    ax1, ax2, ax3 = np.ravel(axes)

    # Plot temperature, ...
    typhon.plots.profile_p_log(data['plev'], data['T'].values.ravel(),
                               ax=ax1, **kwargs)
    ax1.set_xlabel('Temperature [K]')
    ax1.set_xlim(140, 320)

    # ... water vapor ...
    typhon.plots.profile_p_log(data['plev'], data['H2O'].values.ravel(),
                               ax=ax2, **kwargs)
    ax2.set_xlabel('$\mathsf{H_2O}$ [VMR]')
    ax2.set_xlim(0, 0.04)

    typhon.plots.profile_p_log(data['plev'], lw_htngrt,
                               ax=ax3, label='Longwave')
    typhon.plots.profile_p_log(data['plev'], sw_htngrt,
                               ax=ax3, label='Shortwave')
    typhon.plots.profile_p_log(data['plev'], sw_htngrt + lw_htngrt, ax=ax3,
                               label='Net rate', color='k')
    ax3.set_xlabel('Heatingrate [°C/day]')
    ax3.legend(loc='upper center')


def plot_overview_z(data, lw_htngrt, sw_htngrt, axes, **kwargs):
    """Plot overview of atmopsheric temperature and humidity profiles.

    Parameters:
        data:
        lw_htngrt:
        sw_htngrt:
        axes (list, tuple or ndarray): Three AxesSubplots.
        **kwargs: Additional keyword arguments passed to all calls
            of `atmospheric_profile`.
    """
    if len(axes) != 3:
        raise Exception('Need to pass three AxesSubplot.')
    ax1, ax2, ax3 = np.ravel(axes)

    # Plot temperature, ...
    typhon.plots.profile_z(data['Z'], data['T'], ax=ax1, **kwargs)
    ax1.set_xlabel('Temperature [K]')
    ax1.set_xlim(140, 320)

    # ... water vapor ...
    typhon.plots.profile_z(data['Z'], data['H2O'], ax=ax2, **kwargs)
    ax2.set_xlabel('$\mathsf{H_2O}$ [VMR]')
    ax2.set_xlim(0, 0.04)

    typhon.plots.profile_z(data['Z'], lw_htngrt, ax=ax3, label='Longwave')
    typhon.plots.profile_z(data['Z'], sw_htngrt, ax=ax3, label='Shortwave')
    typhon.plots.profile_z(data['Z'], sw_htngrt + lw_htngrt, ax=ax3,
                           label='Net rate', color='k')
    ax3.set_xlabel('Heatingrate [°C/day]')
    ax3.legend(loc='upper center')
