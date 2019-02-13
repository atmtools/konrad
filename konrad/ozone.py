# -*- coding: utf-8 -*-
"""This module contains classes handling different treatments of ozone."""

import os
import abc
import logging
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from konrad.component import Component

__all__ = [
    'Ozone',
    'OzonePressure',
    'OzoneHeight',
    'OzoneNormedPressure',
    'OzoneCariolle',
    'OzoneSiRaChA',
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


class OzoneCariolle(Ozone):
    """Implementation of the Cariolle ozone scheme for the tropics.
    """
    def __init__(self, w=0):
        """
        Parameters:
            w (ndarray / int / float): upwelling velocity [mm / s]
        """
        super().__init__()
        self.w = w * 86.4  # in m / day

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
        if self.w == 0:
            return np.zeros(len(z))

        if isinstance(self.w, np.ndarray):
            w_array = self.w
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
            w_array = w*w_factor

        do3dz = (o3[1:] - o3[:-1]) / np.diff(z)
        do3dz = np.hstack(([0], do3dz))

        return -w_array * do3dz

    def get_params(self, p):
        cariolle_data = Dataset(
            os.path.join(os.path.dirname(__file__),
                         '../Cariolle_data.nc'))
        p_data = cariolle_data['p'][:]
        alist = []
        for param_num in range(1, 8):
            a = cariolle_data[f'A{param_num}'][:]
            alist.append(interp1d(p_data, a, fill_value='extrapolate')(p))
        return alist

    def __call__(self, atmosphere, convection, timestep, *args, **kwargs):

        from SiRaChA.utils import overhead_molecules

        T = atmosphere['T'][0, :]
        p = atmosphere['plev']  # [Pa]
        phlev = atmosphere['phlev']
        o3 = atmosphere['O3'][0, :]  # moles of ozone / moles of air
        z = atmosphere['z'][0, :]  # m

        o3col = overhead_molecules(o3, p, phlev, z, T
                                   ) * 10 ** -4  # in molecules / cm2

        A1, A2, A3, A4, A5, A6, A7 = self.get_params(p)
        # A7 is in molecules / cm2
        # tendency of ozone volume mixing ratio per second
        do3dt = A1 + A2*(o3 - A3) + A4*(T - A5) + A6*(o3col - A7)

        # transport term
        transport_ox = self.ozone_transport(o3, z, convection)

        atmosphere['O3'] = (
            ('time', 'plev'),
            (o3 + (do3dt * 24 * 60**2 + transport_ox) * timestep).reshape(1, -1)
        )


class OzoneSiRaChA(OzoneCariolle):

    def __init__(self, w=0):
        """
        Parameters:
            w (ndarray / int / float): upwelling velocity [mm / s]
        """
        super().__init__()

        from SiRaChA import SiRaChA

        self.w = w * 86.4  # in m / day
        self._ozone = SiRaChA()

    def __call__(self, atmosphere, convection, timestep, zenith, *args,
                 **kwargs):

        o3 = atmosphere['O3'][-1, :]
        z = atmosphere['z'][-1, :]
        p, phlev = atmosphere['plev'], atmosphere['phlev']
        T = atmosphere['T'][-1, :]
        source, sink_ox, sink_nox, sink_hox = self._ozone.tendencies(
            z, p, phlev, T, o3, zenith)
        transport_ox = self.ozone_transport(o3, z, convection)
        do3dt = source - sink_ox - sink_nox + transport_ox - sink_hox

        atmosphere['O3'] = (
            ('time', 'plev'),
            (o3 + do3dt * timestep).reshape(1, -1)
        )

        for term, tendency in [('ozone_source', source),
                               ('ozone_sink_ox', sink_ox),
                               ('ozone_sink_nox', sink_nox),
                               ('ozone_transport', transport_ox),
                               ('ozone_sink_hox', sink_hox)
                               ]:
            if term in self.data_vars:
                self.set(term, tendency)
            else:
                self.create_variable(term, tendency)
