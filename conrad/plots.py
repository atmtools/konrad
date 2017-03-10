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


def plot_overview(p, vmr, T, fig, **kwargs):
    """Plot overview of atmopsheric temperature and humidity profiles.

    Parameters:
        p (ndarray): Pressure [Pa].
        vmr (ndarray): Pressure [Pa].
        T (ndarray): Pressure [Pa].
        fig (Figure): Matplotlib figure with three AxesSubplots.
        **kwargs: Additional keyword arguments passed to all calls
            of `atmospheric_profile`.
    """
    # Plot temperature, ...
    atmospheric_profile(p, T, ax=fig.axes[0], **kwargs)
    fig.axes[0].set_xlabel('Temperaure [K]')
    fig.axes[0].set_xlim(180, 320)

    # ... water vapor ...
    atmospheric_profile(p, vmr, ax=fig.axes[1], **kwargs)
    fig.axes[1].set_xlabel('$\mathsf{H_2O}$ [VMR]')
    fig.axes[1].set_xlim(0, 0.04)

    # ... and relative humidity.
    rh = typhon.atmosphere.relative_humidity(vmr, p, T)
    atmospheric_profile(p, rh, ax=fig.axes[2], **kwargs)
    fig.axes[2].set_xlabel('Relative Humidity')
    fig.axes[2].xaxis.set_major_formatter(_percent_formatter)
    fig.axes[2].set_xlim(0, 1.05)
