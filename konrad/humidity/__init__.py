"""This module contains classes for handling humidity."""
import logging

from konrad.component import Component
from konrad.physics import (relative_humidity2vmr, vmr2relative_humidity)
from .stratosphere import *
from .relative_humidity import *


logger = logging.getLogger(__name__)


class FixedRH(Component):
    """Preserve the relative humidity profile under temperature changes."""
    def __init__(self, rh_func=None, stratosphere_coupling=None):
        """Create a humidity handler.

        Parameters:
            rh_func (callable): Callable that describes the vertical
                relative humidity distribution.
            stratosphere_coupling (callable):
        """
        if stratosphere_coupling is None:
            self._stratosphere_coupling = ColdPointCoupling()
        else:
            self._stratosphere_coupling = stratosphere_coupling

        self._rh_func = rh_func
        self._rh_profile = None

    @property
    def rh_func(self):
        return type(self._rh_func).__name__

    @property
    def stratosphere_coupling(self):
        return type(self._stratosphere_coupling).__name__

    def adjust_humidity(self, atmosphere, **kwargs):
        """Determine the humidity profile based on atmospheric state.

        Parameters:
            TODO: Write docstring.

        Returns:
            ndarray: Water vapor profile [VMR].
        """
        if self._rh_func is not None:
           rh_profile = self._rh_func(atmosphere, **kwargs)
        else:
            if self._rh_profile is None:
                self._rh_profile = vmr2relative_humidity(
                        vmr=atmosphere['H2O'][-1],
                        pressure=atmosphere['plev'],
                        temperature=atmosphere['T'][-1]
                    )
            rh_profile = self._rh_profile

        atmosphere['H2O'][-1, :] = relative_humidity2vmr(
            relative_humidity=rh_profile,
            pressure=atmosphere['plev'],
            temperature=atmosphere['T'][-1]

        )
        self._stratosphere_coupling.adjust_stratospheric_vmr(atmosphere)


class FixedVMR(Component):
    """Keep the water vapor volume mixing ratio constant."""
    def adjust_humidity(self, atmosphere, **kwargs):
        return
