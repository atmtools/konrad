# -*- coding: utf-8 -*-
"""This module contains classes handling different treatments of ozone."""

import abc
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from konrad import constants
from konrad.component import Component

__all__ = [
    'Ozone',
    'OzonePressure',
    'OzoneHeight',
    'OzoneNormedPressure',
]

logger = logging.getLogger(__name__)


class Ozone(Component, metaclass=abc.ABCMeta):
    """Base class to define abstract methods for ozone treatments."""

    def __init__(self):
        """
        Parameters:
            initial_ozone (ndarray): initial ozone vmr profile
        """
        self['initial_ozone'] = (('plev',), None)

    @abc.abstractmethod
    def __call__(self, atmosphere, convection, timestep, zenith):
        """Updates the ozone profile within the atmosphere class.

        Parameters:
            atmosphere (konrad.atmosphere): atmosphere model containing ozone
                concentration profile, height, temperature, pressure and half
                pressure levels at the current timestep
            convection (konrad.convection): convection scheme
            timestep (float): timestep of run [days]
            zenith (float): solar zenith angle,
                angle of the Sun to the vertical [degrees]
        """


class OzonePressure(Ozone):
    """Ozone fixed with pressure, no adjustment needed."""
    def __call__(self, **kwargs):
        return


class OzoneHeight(Ozone):
    """Ozone fixed with height."""
    def __init__(self):
        self._f = None

    def __call__(self, atmosphere, **kwargs):
        if self._f is None:
            self._f = interp1d(
                atmosphere['z'][0, :],
                atmosphere['O3'],
                fill_value='extrapolate',
            )
        atmosphere['O3'] = (('time', 'plev'), self._f(atmosphere['z'][0, :]))


class OzoneNormedPressure(Ozone):
    """Ozone shifts with the normalisation level (chosen to be the convective
    top)."""
    def __init__(self, norm_level=None):
        """
        Parameters:
            norm_level (float): pressure for the normalisation
                normally chosen as the convective top pressure at the start of
                the simulation [Pa]
        """
        self.norm_level = norm_level
        self._f = None

    def __call__(self, atmosphere, convection, **kwargs):
        if self.norm_level is None:
            self.norm_level = convection.get('convective_top_plev')[0]
            # TODO: what if there is no convective top

        if self._f is None:
            self._f = interp1d(
                atmosphere['plev'] / self.norm_level,
                atmosphere['O3'][0, :],
                fill_value='extrapolate',
            )

        norm_new = convection.get('convective_top_plev')[0]

        atmosphere['O3'] = (
            ('time', 'plev'),
            self._f(atmosphere['plev'] / norm_new).reshape(1, -1)
        )


class Ozone_Cariolle(Ozone):
    """Implementation of the Cariolle ozone scheme for the tropics.
    """
    def get_params(self, p):
        param_path = '/home/mpim/m300580/Documents/Cariolle/'
        p_data = pd.read_csv(param_path+'cariolle_plev.dat',
                             delimiter='\n').values.reshape(91,) * 100 # in Pa
        Alist = []
        for param_num in range(1, 8):
            a = pd.read_csv(param_path+'cariolle_a{}.dat'.format(param_num),
                             delimiter='\n').values
            a = a.reshape((65, 91))  # latitude coords: 65, pressure levels: 91
            a = np.mean(a[29:36, :], axis=0)  # mean 10N-10S
            Alist.append(interp1d(p_data, a, fill_value='extrapolate')(p))
        return Alist

    def __call__(self, atmosphere, timestep, *args, **kwargs):

        Rd = constants.molar_Rd  # TODO: should this be for moist air instead?
        T = atmosphere['T'].values[0, :]
        p = atmosphere['plev'].values  # [Pa]
        phlev = atmosphere['phlev'].values
        o3 = atmosphere['O3'].values[0, :]  # moles of ozone / moles of air
        z = atmosphere['z'].values[0, :]  # [m]
        mol_air = p / (Rd * T) * z  # moles / m2 of air
        # o3col: overhead column ozone in moles / m2
        o3col_phlev = np.hstack((np.cumsum((o3 * mol_air)[::-1])[::-1], [0]))
        o3col = interp1d(phlev, o3col_phlev)(p)

        A1, A2, A3, A4, A5, A6, A7 = self.get_params(p)

        # tendency of ozone volume mixing ratio per second
        #TODO: o3col and A7 are NOT in the same units
        do3dt = A1 + A2*(o3 - A3) + A4*(T - A5) + A6*(o3col - A7)

        atmosphere['O3'] = (
            ('time', 'plev'),
            (o3 + do3dt * timestep * 24 * 60**2).reshape(1, -1)
        )
