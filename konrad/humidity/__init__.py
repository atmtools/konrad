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
                If `None`, assume a :class:`HeightConstant` relative humidity.
            stratosphere_coupling (callable): Callable that describes how the
                humidity should be treated in the stratosphere.
        """
        if stratosphere_coupling is None:
            self._stratosphere_coupling = ColdPointCoupling()
        else:
            self._stratosphere_coupling = stratosphere_coupling

        if rh_func is None:
            self._rh_func = HeightConstant()
        else:
            self._rh_func = rh_func

        self._rh_profile = None

    @property
    def netcdf_subgroups(self):
        return {
            'rh_func': self._rh_func,
            'stratosphere_coupling': self._stratosphere_coupling,
        }

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
        atmosphere['H2O'][-1, :] = relative_humidity2vmr(
            relative_humidity=self._rh_func(atmosphere, **kwargs),
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
