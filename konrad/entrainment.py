"""This module contains classes for an entrainment induced cooling term.
"""
import abc

from konrad.component import Component


class Entrainment(Component, metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all entrainment handlers."""

    @abc.abstractmethod
    def entrain(self, T, atmosphere):
        """Entrain air masses to the atmosphere column.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
            timestep (float): Timestep width [day].

        Returns:
            ndarray: Adjusted temperature profile [K].
        """


class NoEntrainment(Entrainment):
    """Do not entrain air."""
    def entrain(self, T, *args, **kwargs):
        return T
