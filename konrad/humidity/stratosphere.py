"""Coupling mechanisms between tropospheric and stratospheric water vapor."""
import abc

import numpy as np

from konrad.component import Component


__al__ = [
    "StratosphereCoupler",
    "ColdPointCoupling",
    "FixedStratosphericVMR",
]


class StratosphereCoupler(Component, metaclass=abc.ABCMeta):
    """Define the coupling of tropospheric and stratospheric water vapor."""

    @abc.abstractmethod
    def adjust_stratospheric_vmr(self, atmosphere):
        """Adjust stratospheric water vapor VMR values.

        Note:
            This method modifies the values in the atmosphere model!

        Parameters:
            atmosphere (``konrad.atmosphere.Atmosphere``):
                Atmosphere component.
        """
        return


class ColdPointCoupling(StratosphereCoupler):
    """Keep stratospheric VMR constant from the cold point on."""

    def adjust_stratospheric_vmr(self, atmosphere):
        cp_index = atmosphere.get_cold_point_index()
        atmosphere["H2O"][-1, cp_index:] = atmosphere["H2O"][-1, cp_index]


class FixedStratosphericVMR(StratosphereCoupler):
    """Keep stratospheric VMR fixed at a constant value."""

    def __init__(self, stratospheric_vmr=5e-6):
        """
        Parameters:
            stratospheric_vmr (float): Stratospheric water vapor amount [VMR].
        """
        self.stratospheric_vmr = stratospheric_vmr

    def adjust_stratospheric_vmr(self, atmosphere):
        cp_index = atmosphere.get_cold_point_index()
        atmosphere["H2O"][-1, cp_index:] = self.stratospheric_vmr
