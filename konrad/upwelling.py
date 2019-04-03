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
from scipy.interpolate import interp1d

from konrad import constants
from konrad.component import Component


def cooling_rates(T, z, w, Cp, base_level):
    """Get cooling rates associated with the upwelling velocity w.

    Parameters:
        T (ndarray): temperature profile [K]
        z (ndarray): height array [m]
        w (int/float/ndarray): upwelling velocity [m/day]
        Cp (int/float/ndarray): Heat capacity [J/K/kg]
        base_level (int): model level index of the base level of the upwelling,
            below this no upwelling is applied
    Returns:
        ndarray: heating rate profile [K/day]
    """
    dTdz = np.gradient(T, z)

    g = constants.g
    Q = -w * (dTdz + g / Cp)
    Q[:base_level] = 0

    return Q


def bdc_profile(norm_level):
    """Define Brewer-Dobson circulation velocity [mm/s] based on the three
    reanalyses shown in Abalos et al. 2015 (doi: 10.1002/2015JD023182)

    Parameters:
        norm_level (float/int): normalisation pressure level [Pa]
    """
    p = np.array([100, 80, 70, 60, 50, 40, 30, 20, 10])*100
    bdc = np.array([0.28, 0.24, 0.23, 0.225, 0.225, 0.24, 0.27, 0.32, 0.42]
                   )*86.4  # [m / day]
    f = interp1d(np.log(p/norm_level), bdc,
                 fill_value=(0.42*86.4, 0.28*86.4), bounds_error=False,
                 kind='quadratic')
    return f


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
        self._w = w * 86.4  # in m/day
        self.lowest_level = lowest_level

    def cool(self, atmosphere, convection, timestep):
        """Apply cooling above the convective top (level where the net
        radiative heating becomes small)."""

        T = atmosphere['T'][0, :]
        z = atmosphere['z'][0, :]
        Cp = atmosphere.get_heat_capacity()

        if self.lowest_level is not None:
            above_level_index = self.lowest_level
        else:
            above_level_index = convection.get('convective_top_index')[0]
            if np.isnan(above_level_index):
                # if convection hasn't been applied and a lowest level for the
                # upwelling has not been specified, upwelling is not applied
                return
        above_level_index = int(np.round(above_level_index))

        Q = cooling_rates(T, z, self._w, Cp, above_level_index)

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


class CoupledUpwelling(StratosphericUpwelling):

    def __init__(self, norm_plev=None):

        self._norm_plev = norm_plev
        self._w = None
        self._f = None

    def cool(self, atmosphere, convection, timestep):

        if self._norm_plev is None:  # first time only and if not specified
            above_level_index = convection.get('convective_top_index')[0]
            if np.isnan(above_level_index):
                # TODO: what if there is no convective top
                return
            above_level_index = int(np.round(above_level_index))
            self._norm_plev = atmosphere['plev'][above_level_index]

        if self._f is None:  # first time only
            self._f = bdc_profile(self._norm_plev)

        above_level_index = int(np.round(
            convection.get('convective_top_index')[0]))
        norm_plev = atmosphere['plev'][above_level_index]
        self._w = self._f(np.log(atmosphere['plev'] / norm_plev))

        T = atmosphere['T'][0, :]
        z = atmosphere['z'][0, :]
        Cp = atmosphere.get_heat_capacity()
        Q = cooling_rates(T, z, self._w, Cp, above_level_index)

        atmosphere['T'][0, :] += Q * timestep

        if 'w' in self.data_vars:
            self.set('w', self._w)
        else:
            self.create_variable('w', self._w)
