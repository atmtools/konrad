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


class ZeroBuoyancyEntrainingPlume(Entrainment):
    """Zero-buoyancy entraining plume with a height-dependent weighting coefficient.

    Adjustment with a lapse rate affected by entrainment between the cloud base
    (960hPa to convective top). Following moist-adiabat at the upper (T_con=T_rad)
    and lower boundaries (surface). Deviating from moist-adiabat in between.
    Initial temperature reduction from entrainment following the zero-buoyancy
    entraining plume model as described by Singh&O'Gorman (2013). Applying a
    height-dependent coefficient to the initial ttemperature reduction to mimick
    the buoyancy-sorting effect of convection as described in Bao et al. (submitted).
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

        entr = self.entr
        deltaT = np.ones_like(p) * 0.0
        k_cb = np.max(np.where(p >= 96000.0))

        # First calculate temperature deviation based on Eq. (4) in Singh&O'Gorman (2013)
        deltaT[k_cb:] = (
            1
            / (1 + L / (Rv * T_con_adiabat[k_cb:] ** 2) * L * q_saturated[k_cb:] / Cp)
            * np.cumsum(
                entr
                / zhlev[k_cb:]
                * (1 - RH_hlev[k_cb:])
                * L
                / Cp
                * q_saturated_hlev[k_cb:]
                * dz_lapse[k_cb:]
            )
        )
        # Second weight deltaT obtained from above by a height-dependent coefficient,
        # as described in Eq. (4) in Bao et al. (submitted).
        if np.any(T_con_adiabat > T_rad):
            k_ttl = np.max(np.where(T_con_adiabat > T_rad))
            z_ttl = z[k_ttl]
            z_cb = z[k_cb]

            f = lambda x: x ** (2.0 / 3.0)
            weight = f((z[k_cb : k_ttl + 1] - z_ttl) / (z_cb - z_ttl))
            deltaT[k_cb : k_ttl + 1] = deltaT[k_cb : k_ttl + 1] * weight
            deltaT[k_ttl + 1 :] = 0

        self.create_variable(
            name="entrainment_cooling",
            dims=("time", "plev"),
            data=deltaT.reshape(1, -1),
        )

        return T_con_adiabat - deltaT
