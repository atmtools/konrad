# -*- coding: utf-8 -*-
"""Plotting related functions.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


__all__ = [
    'atmospheric_profile_p',
    'atmospheric_profile_z',
    'plot_overview_p',
    'plot_overview_z',
]


@FuncFormatter
def _pres_formatter(x, pos):
    return '{:.0f}'.format(x / 1e2)


@FuncFormatter
def _percent_formatter(x, pos):
    return '{:.0f}\N{SIX-PER-EM SPACE}%'.format(x * 100)


def atmospheric_profile_p(p, x, ax=None, **kwargs):
    """Plot atmospheric profile of arbitrary property.

    Parameters:
        p (ndarray): Pressure [Pa].
        x (ndarray): Atmospheric property.
        ax (AxesSubplot): Axes to plot in.
        **kwargs: Additional keyword arguments passed to `plt.plot`.
    """
    if ax is None:
        ax = plt.gca()

    # Determine min/max pressure of **all** data in plot.
    pmin = np.min((p.min(), *ax.get_ylim()))
    pmax = np.max((p.max(), *ax.get_ylim()))
    ax.set_ylim(pmax, pmin)  # implicitly invert yaxis

    # Only plot bottom and left spines.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Label and format for yaxis.
    ax.yaxis.set_major_formatter(_pres_formatter)
    if ax.is_first_col():
        ax.set_ylabel('Pressue [hPa]')

    # Actual plot.
    return ax.plot(x, p, **kwargs)


def atmospheric_profile_z(z, x, ax=None, **kwargs):
    """Plot atmospheric profile of arbitrary property.

    Parameters:
        z (ndarray): Height [m].
        x (ndarray): Atmospheric property.
        ax (AxesSubplot): Axes to plot in.
        **kwargs: Additional keyword arguments passed to `plt.plot`.
    """
    if ax is None:
        ax = plt.gca()

    z = z / 1e3  # scale to km.

    # Determine min/max pressure of **all** data in plot.
    zmin = np.min((z.min(), *ax.get_ylim()))
    zmax = np.max((z.max(), *ax.get_ylim()))
    ax.set_ylim(zmin, zmax)  # implicitly invert yaxis

    # Only plot bottom and left spines.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Label and format for yaxis.
    if ax.is_first_col():
        ax.set_ylabel('Height [km]')

    # Actual plot.
    return ax.plot(x, z, **kwargs)


def plot_overview_p(data, lw_htngrt, sw_htngrt, axes, **kwargs):
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
    atmospheric_profile_p(data.index, data['T'], ax=ax1, **kwargs)
    ax1.set_xlabel('Temperaure [K]')
    ax1.set_xlim(140, 320)

    # ... water vapor ...
    atmospheric_profile_p(data.index, data['Q'], ax=ax2, **kwargs)
    ax2.set_xlabel('$\mathsf{H_2O}$ [VMR]')
    ax2.set_xlim(0, 0.04)

    atmospheric_profile_p(data.index, lw_htngrt, ax=ax3, label='Longwave')
    atmospheric_profile_p(data.index, sw_htngrt, ax=ax3, label='Shortwave')
    atmospheric_profile_p(data.index, sw_htngrt + lw_htngrt, ax=ax3,
                          label='Net rate', color='k')
    ax3.set_xlabel('Heatingrate [°C/day]')
    ax3.set_xlim(-5, 2)
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
    atmospheric_profile_z(data['Z'], data['T'], ax=ax1, **kwargs)
    ax1.set_xlabel('Temperaure [K]')
    ax1.set_xlim(140, 320)

    # ... water vapor ...
    atmospheric_profile_z(data['Z'], data['Q'], ax=ax2, **kwargs)
    ax2.set_xlabel('$\mathsf{H_2O}$ [VMR]')
    ax2.set_xlim(0, 0.04)

    atmospheric_profile_z(data['Z'], lw_htngrt, ax=ax3, label='Longwave')
    atmospheric_profile_z(data['Z'], sw_htngrt, ax=ax3, label='Shortwave')
    atmospheric_profile_z(data['Z'], sw_htngrt + lw_htngrt, ax=ax3,
                          label='Net rate', color='k')
    ax3.set_xlabel('Heatingrate [°C/day]')
    ax3.set_xlim(-5, 2)
    ax3.legend(loc='upper center')
