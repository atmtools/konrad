"""Coupling mechanisms between tropospheric and stratospheric water vapor."""
import abc

import numpy as np

from konrad.component import Component


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
        atmosphere['H2O'][-1, cp_index:] = atmosphere['H2O'][-1, cp_index]


class NonIncreasing(StratosphereCoupler):
    """Prevent the VMR from increasing with height anywhere in the column."""
    def adjust_stratospheric_vmr(self, atmosphere):
        h2o = atmosphere['H2O'][-1, :]
        h2o_grad = np.diff(h2o)
        if not np.all(h2o_grad < 0):
            index = np.argmax(h2o_grad > 0)
            atmosphere['H2O'][-1, index+1:] = atmosphere['H2O'][-1, index]


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
        atmosphere['H2O'][-1, cp_index:] = self.stratospheric_vmr


class MinimumStratosphericVMR(StratosphereCoupler):
    """Prevent stratospheric VMR from decreasing below a fixed minimum value."""
    def __init__(self, minimum_vmr=5e-6):
        """
        Parameters:
            minimum_vmr (float): Minimum water vapor amount [VMR].
        """
        self.minimum_vmr = minimum_vmr

    def adjust_stratospheric_vmr(self, atmosphere):
        vmr = atmosphere['H2O'][-1, :]

        # If the VMR falls below the stratospheric background...
        if np.any(vmr < self.minimum_vmr):
            # ... set all values equal to the background from there.
            vmr[np.argmax(vmr < self.minimum_vmr):] = self.minimum_vmr
        else:
            # Otherwise find the smallest VMR and use the background value
            # from there on, this at least minimizes the discontinuity at
            # the transition point.
            vmr[np.argmin(vmr):] = self.minimum_vmr
