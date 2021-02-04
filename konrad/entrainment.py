"""This module contains classes for an entrainment induced cooling term.
"""
import abc

import numpy as np
from scipy.interpolate import interp1d
from typhon.physics import vmr2mixing_ratio

from konrad import constants
from konrad.component import Component
from konrad.physics import saturation_pressure, vmr2relative_humidity


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


class ZeroBuoyancyEtrainingPlume(Entrainment):
    """Zero-buoyancy entraining plume.

    Adjustment with a lapse rate affected by entrainment below the freezing level.
    Following moist-adiabat at the upper (T_con=T_rad)  and lower boundaries (surface).
    Reduced temperature in between, minimum at freezing level (273.15 kelvin).
    Temperature reduction from entrainment following the zero-buoyancy entraining plume
    model as described by Singh&O'Gorman (2013).
    """

    def __init__(self, entr=0.5):
        """Initialize the entrainment component.

        entr (float): Entrainment parameter.
        """
        self.entr = entr

    def entrain(self, T_con_adiabat, atmosphere):
        # Physical constants.
        L = constants.heat_of_vaporization
        Rv = constants.specific_gas_constant_water_vapor
        Cp = constants.isobaric_mass_heat_capacity_dry_air

        # Abbreviated variables references.
        T_rad = atmosphere["T"][0, :]
        p = atmosphere["plev"][:]
        phlev = atmosphere["phlev"][:]

        # Zero-buoyancy plume entrainment.
        k_ttl = np.max(np.where(T_con_adiabat >= T_rad))
        r_saturated = np.ones_like(p) * 0.0
        r_saturated[: k_ttl + 1] = vmr2mixing_ratio(
            saturation_pressure(T_con_adiabat[: k_ttl + 1]) / p[: k_ttl + 1]
        )
        q_saturated = r_saturated / (1 + r_saturated)
        q_saturated_hlev = interp1d(np.log(p), q_saturated, fill_value="extrapolate")(
            np.log(phlev[:-1])
        )

        z = atmosphere["z"][0, :]
        zhlev = interp1d(np.log(p), z, fill_value="extrapolate")(np.log(phlev[:-1]))
        dz_lapse = np.hstack((np.array([z[0] - zhlev[0]]), np.diff(z)))

        RH = vmr2relative_humidity(atmosphere["H2O"][0, :], p, atmosphere["T"][0, :])
        RH = np.where(RH > 1, 1, RH)
        RH_hlev = interp1d(np.log(p), RH, fill_value="extrapolate")(np.log(phlev[:-1]))

        deltaT = np.ones_like(p) * 0.0
        k_cb = np.max(np.where(p >= 96000.0))
        deltaT[k_cb:] = (
            1
            / (1 + L / (Rv * T_con_adiabat[k_cb:] ** 2) * L * q_saturated[k_cb:] / Cp)
            * np.cumsum(
                self.entr
                / zhlev[k_cb:]
                * (1 - RH_hlev[k_cb:])
                * L
                / Cp
                * q_saturated_hlev[k_cb:]
                * dz_lapse[k_cb:]
            )
        )

        k_fl = np.max(np.where(T_con_adiabat - deltaT >= 273.15))
        p_fl = p[k_fl]
        p_ttl = p[k_ttl]

        f = lambda x: x ** (1.0 / 1.5)
        weight = f((p[k_fl : k_ttl + 1] - p_ttl) / (p_fl - p_ttl))
        deltaT[k_fl : k_ttl + 1] = deltaT[k_fl : k_ttl + 1] * weight
        deltaT[k_ttl + 1 :] = 0

        self.create_variable(
            name="entrainment_cooling",
            dims=("time", "plev"),
            data=deltaT.reshape(1, -1),
        )

        return T_con_adiabat - deltaT
