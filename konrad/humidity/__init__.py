"""This module contains classes for handling humidity."""
import logging

from konrad.component import Component
from konrad.utils import prefix_dict_keys
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
    def attrs(self):
        # Overrides ``Component.attrs`` by returning a composite of the
        # attribtues of both ``rh_func`` and ``stratosphere_coupling``.
        # The returned attributes are prefixed with their parent-attribute's
        # name for clarity.
        attrs = dict(
            **prefix_dict_keys(
                getattr(self._rh_func, 'attrs', {}), 'rh_func'
            ),
            **prefix_dict_keys(
                self._stratosphere_coupling.attrs, 'stratosphere_coupling',
            )
        )
        attrs['rh_func/class'] = self.rh_func
        attrs['stratosphere_coupling/class'] = self.stratosphere_coupling

        return attrs

    def hash_attributes(self):
        # Make sure that non-``Component`` attributes do not break hashing.
        return hash(tuple(
            attr.hash_attributes()
            for attr in (self._rh_func, self._stratosphere_coupling)
            if hasattr(attr, 'hash_attributes')
        ))

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
    def __init__(self, *args, **kwargs):
        if len(args) + len(kwargs) > 0:
            # Allow arguments to be passed for consistent interface but
            # warn the user.
            logger.warning(f'All input arguments to {self} are ignored.')

        # Set both attributes for consistent user interface and netCDF output.
        self.rh_func = 'FixedVMR'
        self.stratosphere_coupling = 'FixedVMR'

    def adjust_humidity(self, atmosphere, **kwargs):
        return
