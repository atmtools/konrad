# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from typhon import atmosphere
from . import utils
from .plots import atmospheric_profile_z
import matplotlib.pyplot as plt


__all__ = [
    'ConRad',
]


class ConRad():
    """Implementation of a radiative-convective equilibrium model."""
    def __init__(self, sounding=None, adjust_vmr=True, dt=1,
                 max_iterations=365, delta=0.07, plot_iterations=True):
        """Set-up a radiative-convective model.

        Parameters:
            sounding (pd.DataFrame): pandas DataFrame representing an
                atmospheric sounding.
            adjust_vmr (bool): Adjust the water vapor mixing ratio to keep
                the relative humidity constant.
            dt (float): Time step in days.
            max_iterations (int): Maximum number of iterations.
            delta (float): Stop criterion. If the heating rate is below this
                threshold for all levels, skip further iterations.
            plot_iterations (bool): Plot iterations.
        """
        self.sounding = sounding
        self.adjust_vmr = adjust_vmr
        self.dt = dt
        self.max_iterations = max_iterations
        self.niter = 0
        self.delta = delta
        self.rad_lw = None
        self.rad_sw = None
        self.plot_iterations = plot_iterations

    def plot_overview_z(self, fig, **kwargs):
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
        atmospheric_profile_z(self.sounding['Z'], self.sounding['T'],
                              ax=fig.axes[0], **kwargs)
        fig.axes[0].set_xlabel('Temperaure [K]')
        fig.axes[0].set_xlim(140, 320)

        # ... water vapor ...
        atmospheric_profile_z(self.sounding['Z'], self.sounding['Q'] / 1000,
                              ax=fig.axes[1], **kwargs)
        fig.axes[1].set_xlabel('$\mathsf{H_2O}$ [VMR]')
        fig.axes[1].set_xlim(0, 0.04)

        atmospheric_profile_z(self.sounding['Z'], self.rad_lw['lw_htngrt'],
                              ax=fig.axes[2], label='Longwave')
        atmospheric_profile_z(self.sounding['Z'], self.rad_sw['sw_htngrt'],
                              ax=fig.axes[2], label='Shortwave')
        atmospheric_profile_z(
            self.sounding['Z'],
            self.rad_sw['sw_htngrt'] + self.rad_lw['lw_htngrt'],
            ax=fig.axes[2], label='Net rate', color='k')
        fig.axes[2].set_xlabel('Heatingrate [°C/day]')
        fig.axes[2].set_xlim(-5, 2)
        fig.axes[2].legend(loc='upper center')

        fig.axes[0].set_ylim(0, 30)
        fig.axes[0].set_title('Iteration: {}'.format(self.niter))
        fig.axes[1].set_ylim(0, 30)
        fig.axes[2].set_ylim(0, 30)

    @utils.with_psrad_symlinks
    def run(self):
        """Run the radiative-convective equilibirum model."""
        from . import psrad

        while self.niter < self.max_iterations:
            self.rad_lw = psrad.psrad_lw(self.sounding)
            self.rad_sw = psrad.psrad_sw(self.sounding)
            net_rate = (self.rad_lw['lw_htngrt'] + self.rad_sw['sw_htngrt'])

            T_new = self.sounding['T'] + net_rate

            if self.adjust_vmr:
                rh = atmosphere.relative_humidity(
                    self.sounding['Q'] / 1000,
                    self.sounding['P'],
                    self.sounding['T'])

                self.sounding['Q'] = atmosphere.vmr(
                    rh, self.sounding['P'], T_new) * 1000

            self.sounding['T'] = T_new

            if self.plot_iterations:
                fig, axes = plt.subplots(1, 3, sharey=True, figsize=(8, 6))
                self.plot_overview_z(fig)
                fig.savefig('plots/iter_{:03d}.png'.format(self.niter),
                            bbox_inches='tight')
                plt.close(fig)

            if np.all(np.abs(net_rate) < self.delta):
                break

            print('Iteration {}...'.format(self.niter))
            self.niter += 1

