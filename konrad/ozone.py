# -*- coding: utf-8 -*-
"""This module contains classes handling different treatments of ozone.
The ozone models can either be used in RCE simulations, or it may be of
interest to run the :py:class:`Cariolle` or :py:class:`Simotrostra` models with
a fixed atmospheric temperature profile.

**In an RCE simulation**

Create an instance of an ozone class, *e.g.* :py:class:`OzoneHeight`,
and use it in an RCE simulation:

    >>> import konrad
    >>> ozone_fixed_with_height = konrad.ozone.OzoneHeight()
    >>> rce = konrad.RCE(atmosphere=..., ozone=ozone_fixed_with_height)
    >>> rce.run()

**Run an ozone model**

Create an ozone model, *e.g.* :py:class:`Cariolle`, and run the ozone model for
a fixed temperature profile:

    >>> import konrad
    >>> ozone_model = konrad.ozone.Cariolle(w=...)
    >>> atmosphere = konrad.atmosphere.Atmosphere(plev=...)
    >>> atmosphere['T'][0] = ...  # set the temperature profile
    >>> atmosphere['O3'][0] = ...  # set the initial ozone profile
    >>> for iteration in range(0, ...):
    >>>     ozone_model(atmosphere=atmosphere, timestep=...)
    >>> final_ozone_profile = atmosphere['O3'][-1]

Note that to use :py:class:`Cariolle` and :py:class:`Simotrostra`, the
simotrostra package needs to be installed. It can be found here:
https://gitlab.com/simple-stratospheric-chemistry/simotrostra

"""

import os
import abc
import logging
import numpy as np
import numbers
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from konrad.component import Component

__all__ = [
    'Ozone',
    'OzonePressure',
    'OzoneHeight',
    'OzoneNormedPressure',
    'Cariolle',
    'Simotrostra',
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
            convection (konrad.convection): convection model containing
                information about the convective top
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
    """Ozone shifts with the normalisation level."""
    def __init__(self, norm_level=None, coupling="convective_top"):
        """
        Parameters:
            norm_level (float): pressure for the normalisation
                normally chosen as the convective top pressure at the start of
                the simulation [Pa]
            coupling (str): Mechanism to determine the normalisation level:
                    * "convective_top": Coupled to the convective top.
                    * "cold_point": Coupled to the cold point tropopause.
        """
        self.norm_level = norm_level
        self._f = None
        self.coupling = coupling


    def get_norm_level(self, atmosphere, convection):
        if self.coupling == 'convective_top':
            # TODO: what if there is no convective top
            return convection.get('convective_top_plev')[0]
        elif self.coupling == 'cold_point':
            return atmosphere.get_cold_point_plev(interpolate=True)
        else:
            raise ValueError("Invalid coupling mechanism.")

    def __call__(self, atmosphere, convection, **kwargs):
        if self.norm_level is None:
            self.norm_level = self.get_norm_level(atmosphere, convection)

        if self._f is None:
            self._f = interp1d(
                atmosphere['plev'] / self.norm_level,
                atmosphere['O3'][0, :],
                fill_value='extrapolate',
            )

        norm_new = self.get_norm_level(atmosphere, convection)

        atmosphere['O3'] = (
            ('time', 'plev'),
            self._f(atmosphere['plev'] / norm_new).clip(min=0).reshape(1, -1)
        )


class Cariolle(Ozone):
    """Implementation of the Cariolle ozone scheme for the tropics.
    """
    def __init__(self, w=0, is_coupled_upwelling=False):
        """
        Parameters:
            w (ndarray / int / float): upwelling velocity [mm / s]
            is_coupled_upwelling (bool): whether to couple to upwelling model
        """
        super().__init__()
        if not is_coupled_upwelling:
            self.w = w * 86.4  # in m / day
        else:
            self.w = None
        self._is_coupled_upwelling = is_coupled_upwelling

    def ozone_transport(self, o3, z, upwelling):
        """Rate of change of ozone is calculated based on the ozone gradient
        and an upwelling velocity.

        Parameters:
            o3 (ndarray): ozone concentration [ppv]
            z (ndarray): height [m]
            upwelling (konrad.upwelling): upwelling model
        Returns:
            ndarray: change in ozone concentration [ppv / day]
        """
        if self.w == 0:  # no transport
            return np.zeros(len(z))

        if self._is_coupled_upwelling:  # take value from upwelling class
            w_array = upwelling._w
        elif isinstance(self.w, np.ndarray):  # use input array
            w_array = self.w
        elif isinstance(self.w, numbers.Number):  # w is a single value
            # transform to an array to apply transport everywhere
            w = self.w
            numlevels = len(z)
            w_factor = np.ones(numlevels)
            w_array = w*w_factor

        do3dz = np.gradient(o3, z)

        return -w_array * do3dz

    def get_params(self, p):
        cariolle_data = Dataset(
            os.path.join(os.path.dirname(__file__),
                         'data/Cariolle_data.nc'))
        p_data = cariolle_data['p'][:]
        alist = []
        for param_num in range(1, 8):
            a = cariolle_data[f'A{param_num}'][:]
            alist.append(interp1d(p_data, a, fill_value='extrapolate')(p))
        return alist

    def __call__(self, atmosphere, timestep, upwelling, **kwargs):

        try:
            from simotrostra.utils import overhead_molecules
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "No module named 'simotrostra'. You need to install it in order"
                " to use this ozone class.\n"
                "It can be found here:\n"
                "https://gitlab.com/simple-stratospheric-chemistry/simotrostra"
            )

        T = atmosphere['T'][0, :]
        p = atmosphere['plev']  # [Pa]
        o3 = atmosphere['O3'][0, :]  # moles of ozone / moles of air
        z = atmosphere['z'][0, :]  # m

        o3col = overhead_molecules(o3, p, z, T
                                   ) * 10 ** -4  # in molecules / cm2

        A1, A2, A3, A4, A5, A6, A7 = self.get_params(p)
        # A7 is in molecules / cm2
        # tendency of ozone volume mixing ratio per second
        do3dt = A1 + A2*(o3 - A3) + A4*(T - A5) + A6*(o3col - A7)

        # transport term
        transport_ox = self.ozone_transport(o3, z, upwelling)

        atmosphere['O3'] = (
            ('time', 'plev'),
            (o3 + (do3dt * 24 * 60**2 + transport_ox) * timestep).reshape(1, -1)
        )


class Simotrostra(Cariolle):
    """Wrapper for Ed Charlesworth's simple chemistry scheme.
    """
    def __init__(self, w=0, is_coupled_upwelling=False):
        """
        Parameters:
            w (ndarray / int / float): upwelling velocity [mm / s]
            is_coupled_upwelling (bool): whether to couple to upwelling model
        """
        super().__init__(w=w, is_coupled_upwelling=is_coupled_upwelling)

        try:
            from simotrostra import Simotrostra
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "No module named 'simotrostra'. You need to install it in order"
                " to use this ozone class.\n"
                "It can be found here:\n"
                "https://gitlab.com/simple-stratospheric-chemistry/simotrostra"
            )

        self._ozone = Simotrostra()

    def simotrostra_profile(self, o3, atmosphere, timestep, zenith, upwelling):
        """
        Parameters:
            o3 (ndarray): ozone profile
            atmosphere (konrad.atmosphere)
            timestep (float): timestep of run [days]
            zenith (float): solar zenith angle,
                angle of the Sun to the vertical [degrees]
            upwelling (konrad.upwelling): upwelling model
        Returns:
            ndarray: new ozone profile
            list of ndarrays: source and sink terms
        """
        z = atmosphere['z'][-1, :]
        p, phlev = atmosphere['plev'], atmosphere['phlev']
        T = atmosphere['T'][-1, :]
        source, sink_ox, sink_nox, sink_hox = self._ozone.tendencies(
            z, p, phlev, T, o3, zenith)
        transport_ox = self.ozone_transport(o3, z, upwelling)
        do3dt = source - sink_ox - sink_nox + transport_ox - sink_hox
        o3_new = o3 + do3dt*timestep

        # prevent concentrations getting too low - set tropospheric value
        o3_new.clip(min=4e-8, out=o3_new)

        return o3_new, [source, sink_ox, sink_nox, transport_ox, sink_hox]

    def store_sink_terms(self, sink_terms):
        """
        Parameters:
            sink_terms (list of ndarrays): source and sink terms [ppv / day]
        """
        source, sink_ox, sink_nox, transport_ox, sink_hox = sink_terms
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

    def __call__(self, atmosphere, timestep, upwelling, zenith, **kwargs):

        o3 = atmosphere['O3'][-1, :]
        o3_new, sink_terms = self.simotrostra_profile(o3, atmosphere,
                                                      timestep, zenith,
                                                      upwelling)

        atmosphere['O3'] = (('time', 'plev'), o3_new.reshape(1, -1))

        self.store_sink_terms(sink_terms)
