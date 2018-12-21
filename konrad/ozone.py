# -*- coding: utf-8 -*-
"""This module contains classes handling different treatments of ozone."""

import abc
import logging
import numpy as np
from netCDF4 import Dataset
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


def change_in(z):
    """
    Parameters:
        z (ndarray): height values of levels [m]
    Return
        ndarray: change in height between levels [m]
    """
    # TODO: make dz between half levels
    return np.hstack((z[0], np.diff(z)))


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


class Ozone_Scheme(Ozone):
    """
    Define some class functions used by both Ozone_Cariolle and SiRaChA.
    """
    def overhead_molecules(self, gas, p, phlev, dz, density_molecules):
        """
        Parameters:
            gas (ndarray): gas concentration [ppv] corresponding to levels p
            p (ndarray): pressure levels [Pa]
            phlev (ndarray): half pressure levels [Pa]
            z (ndarray): height values [m]
            T (ndarray): temperature values [K]
        Returns:
            ndarray: number of molecules per m2 in overhead column
        """
        # molecules / m2 of air in each layer
        molecules_air = density_molecules * dz
        # overhead column in molecules / m2
        col_phlev = np.hstack(
                (np.flipud(np.cumsum(np.flipud(gas * molecules_air))), [0]))
        col = interp1d(phlev, col_phlev)(p)

        return col

    def density_of_molecules(self, p, T):
        """
        Parameters:
            p (ndarray): pressure levels [Pa]
            T (ndarray): temperature values [K]
        Returns:
            ndarray: density of molecules [number of molecules / m3]
        """
        return (constants.avogadro * p) / (constants.molar_Rd * T)

    def ozone_transport(self, o3, z, convection):
        """Rate of change of ozone is calculated based on the ozone gradient
        and an upwelling velocity.

        Parameters:
            o3 (ndarray): ozone concentration [ppv]
            z (ndarray): height [m]
            convection (konrad.convection): to get the convective top index
        Returns:
            ndarray: change in ozone concentration [ppv / day]
        """
        if isinstance(self.w, np.ndarray):
            w_factor = 1
        else:  # w is a single value
            # apply transport only above convective top
            w = self.w
            numlevels = len(z)
            contopi = convection.get('convective_top_index')[0]
            if np.isnan(contopi):
                # No convective top index found; do not apply transport term
                return np.zeros(numlevels)
            contopi = int(np.round(contopi))
            w_factor = np.ones(numlevels)
            w_factor[:contopi] = 0

        do3dz = (o3[1:] - o3[:-1]) / np.diff(z)
        do3dz = np.hstack(([0], do3dz))

        return -w*w_factor*do3dz


class Ozone_Cariolle(Ozone_Scheme):
    """Implementation of the Cariolle ozone scheme for the tropics.
    """
    def __init__(self, w=0):
        """
        Parameters:
            w (ndarray / int / float): upwelling velocity [mm / s]
        """
        super().__init__()
        self.w = w * 86.4  # in m / day

    def get_params(self, p):
        cariolle_data = Dataset('Cariolle_data.nc')
        p_data = cariolle_data['p'][:]
        alist = []
        for param_num in range(1, 8):
            a = cariolle_data[f'A{param_num}'][:]
            alist.append(interp1d(p_data, a, fill_value='extrapolate')(p))
        return alist

    def __call__(self, atmosphere, convection, timestep, *args, **kwargs):

        T = atmosphere['T'][0, :]
        p = atmosphere['plev']  # [Pa]
        phlev = atmosphere['phlev']
        o3 = atmosphere['O3'][0, :]  # moles of ozone / moles of air
        z = atmosphere['z'][0, :]  # m

        o3col = self.overhead_molecules(o3, p, phlev, change_in(z),
                                   self.density_of_molecules(p, T)
                                   ) * 10 ** -4  # in molecules / cm2

        A1, A2, A3, A4, A5, A6, A7 = self.get_params(p)
        # A7 is in molecules / cm2
        # tendency of ozone volume mixing ratio per second
        do3dt = A1 + A2*(o3 - A3) + A4*(T - A5) + A6*(o3col - A7)

        # transport term
        if self.w != 0:
            transport_ox = self.ozone_transport(o3, z, convection)
        else:
            transport_ox = 0

        atmosphere['O3'] = (
            ('time', 'plev'),
            (o3 + (do3dt * 24 * 60**2 + transport_ox) * timestep).reshape(1, -1)
        )
