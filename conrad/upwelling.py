# -*- coding: utf-8 -*-
"""This module contains classes for an upwelling induced cooling term."""
import abc

import numpy as np
from conrad import constants


class Upwelling(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all humidity handlers."""

    @abc.abstractmethod
    def cool(self, atmosphere, radheat, timestep):
        """ Cool the atmosphere according to an upwelling.
        Parameters:
              atmosphere (conrad.atmosphere.Atmosphere): Atmosphere model.
              radheat (ndarray): Radiative heatingrate [K/day].
              timestep (float): Timestep width [day].
        """

class NoUpwelling(Upwelling):
    """Do not apply cooling."""
    def cool(self, *args, **kwargs):
        pass

class StratosphericUpwelling(Upwelling):

    def __init__(self, w=43.2):
        """Create a upwelling handler.

        Parameters:
            w (float): Upwelling velocity in m/day.
                43.2 is equilvalent to 0.5 mm/s.
        """
        self.w = w

    def cool(self, atmosphere, radheat, timestep):
        """Apply cooling above the convective top (level where the net
        radiative heating becomes small)."""

        # arbitrary value close to 0, not too close to allow the upwelling to
        # occur when the upper atmosphere is not yet in radiative equilibrium.
        radheatmin = 0.0001

        Q = self.coolingrates(atmosphere)
        T = atmosphere['T'][0, :]
        T_new = T + Q * timestep

        try:
            contopi = np.min(np.where(radheat < radheatmin))
        except ValueError:
            # If a ValueError is thrown, no minimum in radiative heating has
            # been found. Return the function without applying any upwelling.
            return

        atmosphere['T'].values[0, contopi:] = T_new[contopi:]

    def coolingrates(self, atmosphere):
        """Get cooling rates associated with the upwelling velocity w."""
        dz = np.diff(atmosphere['z'][0, :])
        dT = np.diff(atmosphere['T'][0, :])

        g = constants.g
        Cp = constants.Cp
        Q = -self.w * (dT/dz + g/Cp)

        # NOTE: This is not properly interpolated and is half a level out.
        return np.hstack((Q[0], Q))
