# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from typhon import atmosphere
from . import psrad


__all__ = [
    'ConRad',
]


class ConRad():
    """Implementation of a radiative-convective equilibrium model."""


    def __init__(self, sounding=None, adjust_vmr=True, dt=1,
            max_iterations=365, delta=0.07):
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
        """
        self.sounding = sounding
        self.adjust_vmr = adjust_vmr
        self.dt = dt
        self.max_iterations = max_iterations
        self.delta = delta
        self.rad_lw = None
        self.rad_sw = None

    def run(self):
        """Run the radiative-convective equilibirum model."""
        n = 0

        while n < self.max_iterations:
            self.rad_lw = psrad.psrad_lw(self.sounding)
            self.rad_sw = psrad.psrad_sw(self.sounding)
            net_rate = (self.rad_lw['lw_htngrt'] + self.rad_sw['sw_htngrt'])

            T_new = self.sounding['T'] + net_rate

            if self.adjust_vmr:
                rh = atmosphere.relative_humidity(
                    self.sounding['Q'], self.sounding['P'], self.sounding['T'])

                self.sounding['Q'] = atmosphere.vmr(rh, self.sounding['P'], T_new)

            self.sounding['T'] = T_new

            if np.all(np.abs(net_rate) < self.delta):
                break

            print(n)
            n += 1
