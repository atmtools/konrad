# -*- coding: utf-8 -*-
"""This module defines methods required for a radiation scheme to be used in
a radiative-convective simulation and also contains a wrapper for RRTMG.

**In an RCE simulation**

Create an instance of the RRTMG class, and use it in an RCE simulation:

    >>> import konrad
    >>> rad = konrad.radiation.RRTMG(...)
    >>> rce = konrad.RCE(atmosphere=..., radiation=rad)
    >>> rce.run()

**Calculating radiative fluxes or heating rates**

The radiation scheme can also be used outside of an RCE simulation,
to calculate cloudy and clear-sky, longwave and shortwave radiative fluxes
and heating rates:

    >>> import konrad
    >>> rad = konrad.radiation.RRTMG(...)
    >>> rad.calc_radiation(atmosphere=...)
    >>> longwave_heating_rate = rad['lw_htngrt'][-1]

"""
from .arts import ARTS
from .radiation import Radiation
from .rrtmg import RRTMG
from .common import *


__all__ = [s for s in dir() if not s.startswith('_')]
