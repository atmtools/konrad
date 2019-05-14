# -*- coding: utf-8 -*-
"""This module contains classes for an upwelling induced cooling term.
To include an upwelling, use :py:class:`StratosphericUpwelling`, otherwise use
:py:class:`NoUpwelling`.

**Example**

Create an instance of the upwelling class, set the upwelling velocity,
and use the upwelling in an RCE simulation.
    >>> import konrad
    >>> stratospheric_upwelling = konrad.upwelling.StratosphericUpwelling(w=...)
    >>> rce = konrad.RCE(atmosphere=..., upwelling=stratospheric_upwelling)
    >>> rce.run()

"""
import abc

import numpy as np

from konrad import constants
from konrad.component import Component


def cooling_rates(T, z, w, base_level):
    """Get cooling rates associated with the upwelling velocity w.

    Parameters:
        T (ndarray): temperature profile [K]
        z (ndarray): height array [m]
        w (int/float/ndarray): upwelling velocity [m/day]
        base_level (int): model level index of the base level of the upwelling,
            below this no upwelling is applied
    Returns:
        ndarray: heating rate profile [K/day]
    """
    dTdz = np.gradient(T, z)

    g = constants.g
    Cp = constants.Cp
    Q = -w * (dTdz + g / Cp)
    Q[:base_level] = 0

    return Q


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

        # get the base level for the upwelling
        if self.lowest_level is not None:
            contopi = self.lowest_level
        else:
            contopi = convection.get('convective_top_index')[0]
            if np.isnan(contopi):
                # if convection hasn't been applied and a lowest level for the
                # upwelling has not been specified, upwelling is not applied
                return
        contopi = int(np.round(contopi))

        T = atmosphere['T'][0, :]
        z = atmosphere['z'][0, :]
        Q = cooling_rates(T, z, self.w, contopi)

        atmosphere['T'][0, :] += Q * timestep


class SpecifiedCooling(Upwelling):
    """Include an upwelling with specified cooling"""
    def __init__(self, Q):
        """
        Parameters:
            Q (ndarray): heating rate profile [K/day]
        """
        self._Q = Q

    def cool(self, atmosphere, timestep, **kwargs):
        atmosphere['T'][0, :] += self._Q * timestep
