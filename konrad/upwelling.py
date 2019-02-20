# -*- coding: utf-8 -*-
"""This module contains classes for an upwelling induced cooling term."""
import abc

import numpy as np

from konrad import constants
from konrad.component import Component


class Upwelling(Component, metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all upwelling handlers."""

    @abc.abstractmethod
    def cool(self, atmosphere, radheat, timestep):
        """ Cool the atmosphere according to an upwelling.
        Parameters:
              atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
              radheat (ndarray): Radiative heatingrate [K/day].
              timestep (float): Timestep width [day].
        """


class NoUpwelling(Upwelling):
    """Do not apply a dynamical cooling."""
    def cool(self, *args, **kwargs):
        pass


class StratosphericUpwelling(Upwelling):
    """Apply a dynamical cooling, based on a specified upwelling velocity."""
    def __init__(self, w=0.2, lowest_level=None):
        """Create a upwelling handler.

        Parameters:
            w (float): Upwelling velocity in mm/s.
            lowest_level (int or None): The index of the lowest level to which
                the upwelling is applied. If none, uses the top of convection.
        """
        self.w = w * 86.4  # in m/day
        self.lowest_level = lowest_level

    def cool(self, atmosphere, convection, timestep):
        """Apply cooling above the convective top (level where the net
        radiative heating becomes small)."""

        Q = self.coolingrates(atmosphere)
        T = atmosphere['T'][0, :]

        if self.lowest_level is not None:
            contopi = self.lowest_level
        else:
            contopi = convection.get('convective_top_index')[0]
            if contopi is np.nan:
                return
        contopi = int(np.round(contopi))
        T_new = T[contopi:] + Q[contopi:] * timestep
        atmosphere['T'][0, contopi:] = T_new

    def coolingrates(self, atmosphere):
        """Get cooling rates associated with the upwelling velocity w."""
        dTdz = np.gradient(atmosphere['T'][0, :], atmosphere['z'][0, :])

        g = constants.g
        Cp = constants.Cp
        Q = -self.w * (dTdz + g/Cp)

        return Q
