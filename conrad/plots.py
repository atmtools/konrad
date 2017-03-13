# -*- coding: utf-8 -*-
"""Plotting related functions.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import typhon


__all__ = [
    'atmospheric_profile',
    'plot_overview',
]


@FuncFormatter
def _pres_formatter(x, pos):
    return '{:.0f}'.format(x / 1e2)


@FuncFormatter
def _percent_formatter(x, pos):
    return '{:.0f}\N{SIX-PER-EM SPACE}%'.format(x * 100)


def atmospheric_profile(p, x, ax=None, **kwargs):
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


def plot_overview(data, rad_lw, rad_sw, fig, **kwargs):
    """Plot overview of atmopsheric temperature and humidity profiles.

    Parameters:
        data:
        rad_lw:
        rad_sw:
        fig (Figure): Matplotlib figure with three AxesSubplots.
        **kwargs: Additional keyword arguments passed to all calls
            of `atmospheric_profile`.
    """
    # Plot temperature, ...
    atmospheric_profile(data.index * 100, data['T'], ax=fig.axes[0], **kwargs)
    fig.axes[0].set_xlabel('Temperaure [K]')
    fig.axes[0].set_xlim(140, 320)

    # ... water vapor ...
    atmospheric_profile(data.index * 100, data['Q'] / 1000, ax=fig.axes[1], **kwargs)
    fig.axes[1].set_xlabel('$\mathsf{H_2O}$ [VMR]')
    fig.axes[1].set_xlim(0, 0.04)

    atmospheric_profile(
        data.index * 100, rad_lw['lw_htngrt'], ax=fig.axes[2], label='Longwave')
    atmospheric_profile(
        data.index * 100, rad_sw['sw_htngrt'], ax=fig.axes[2], label='Shortwave')
    atmospheric_profile(
        data.index * 100, rad_sw['sw_htngrt'] + rad_lw['lw_htngrt'], ax=fig.axes[2],
        label='Net rate', color='k')
    fig.axes[2].set_xlabel('Heatingrate [°C/day]')
    fig.axes[2].set_xlim(-5, 2)
    fig.axes[2].legend(loc='upper center')
