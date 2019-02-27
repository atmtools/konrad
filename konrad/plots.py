# -*- coding: utf-8 -*-
"""Plotting related functions.
"""
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
import typhon.plots


__all__ = [
    'plot_overview_p_log',
    'plot_overview_z',
    'gregory_plot',
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
    typhon.plots.profile_p_log(data['plev'], data['T'].ravel(),
                               ax=ax1, **kwargs)
    ax1.set_xlabel('Temperature [K]')
    ax1.set_xlim(140, 320)

    # ... water vapor ...
    typhon.plots.profile_p_log(data['plev'], data['H2O'].ravel(),
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


def gregory_plot(temperature, forcing, draw_fit=True, ax=None, **kwargs):
    """Gregory plot to estimate climate sensitivity.

    Parameters:
          temperature (ndarray): Surface temperatures [K].
          forcing (ndarray): Radiation budget at top of atmosphere [W//m^2].
          draw_fit (bool): Whether to draw the linear regression.
          ax (AxesSubplot): Axes to plot in.
          **kwargs: Additional keyword arguments are passed to ``plt.plot``.

    Returns:
          float, float: Estimated climate sensitivty [K / (W/m^2)],
            effective forcing [W / m^2].

    Examples:
        >>> temperature = np.linspace(300, 304, 25)
        >>> forcing = np.linspace(3.7, 0., temperature.size)
        >>> gregory_plot(temperature, forcing)
        (1.0810810810810814, 3.6999999999999993)

    """
    # If no axis is passed, use current axis (will create one, if needed).
    if ax is None:
        ax = plt.gca()

    # Default keyword arguments to control the plot appearance later.
    default_kwargs = {
        'marker': 'o',
        'linestyle': 'none',
    }
    default_kwargs.update(kwargs)

    # Convert temperatures into change in temperature.
    t_change = temperature - temperature[0]

    # Plot radiative forcing against surface temperature change.
    line, = ax.plot(t_change, forcing, **default_kwargs)
    ax.set_xlabel('Surface temperature change [K]')
    ax.set_ylabel('Radiation budget TOA [W/m$^2$]')
    ax.grid(True)

    # Find the maximum radiative forcing. This way, possible adjustment
    # processes during the first timesteps are not taken into account.
    max_forcing = np.argmax(forcing)
    sensitivity, eff_forcing = np.polyfit(t_change[max_forcing:],
                                          forcing[max_forcing:], 1)

    # Create x-values for "Gregory" fit. Values are plotted between zero
    # temperature change and zero forcing.
    x = np.linspace(0, - eff_forcing / sensitivity, 10)

    if draw_fit:
        # Plot linear "Gregory" fit into same axis.
        ax.plot(x, eff_forcing + sensitivity * x, color=line.get_color())

    # Return the estimated climate sensivity in units of K / (W/m^2) and the
    # effective forcing in W / m^2.
    return -1 / sensitivity, eff_forcing
