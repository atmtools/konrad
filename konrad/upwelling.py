# -*- coding: utf-8 -*-
"""This module contains classes for an upwelling induced cooling term."""
import abc

import numpy as np
from konrad import constants


class Upwelling(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all humidity handlers."""

    @abc.abstractmethod
    def cool(self, atmosphere, radheat, timestep):
        """ Cool the atmosphere according to an upwelling.
        Parameters:
              atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
              radheat (ndarray): Radiative heatingrate [K/day].
              timestep (float): Timestep width [day].
        """

class NoUpwelling(Upwelling):
    """Do not apply cooling."""
    def cool(self, *args, **kwargs):
        pass

class StratosphericUpwelling(Upwelling):

    def __init__(self, w=0.2, lowest_level=None):
        """Create a upwelling handler.

        Parameters:
            w (float): Upwelling velocity in mm/s.
            lowest_level (int or None): The index of the lowest level to which
                the upwelling is applied. If none, uses the top of convection.
        """
        self.w = w * 86.4 # in m/day
        self.lowest_level = lowest_level

    def cool(self, atmosphere, radheat, timestep):
        """Apply cooling above the convective top (level where the net
        radiative heating becomes small)."""

        Q = self.coolingrates(atmosphere)
        T = atmosphere['T'][0, :]

        if self.lowest_level is not None:
            contopi = self.lowest_level
        else:
            # arbitrary value close to 0, not too close to allow the upwelling to
            # occur when the upper atmosphere is not yet in radiative equilibrium.
            radheatmin = 0.0001
            try:
                contopi = np.min(np.where(radheat[0, :] > -radheatmin))
            except ValueError:
                # If a ValueError is thrown, no minimum in radiative heating has
                # been found. Return the function without applying any upwelling.
                return
        T_new = T[contopi:] + Q[contopi:] * timestep
        atmosphere['T'][0, contopi:] = T_new

    def coolingrates(self, atmosphere):
        """Get cooling rates associated with the upwelling velocity w."""
        dz = np.diff(atmosphere['z'][0, :])
        dT = np.diff(atmosphere['T'][0, :])

        g = constants.g
        Cp = constants.Cp
        Q = -self.w * (dT/dz + g/Cp)

        # NOTE: This is not properly interpolated and is half a level out.
        return np.hstack((Q[0], Q))
