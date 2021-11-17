"""Different models to describe the vertical relative humidity distribution."""
import abc

import numpy as np
from scipy.interpolate import interp1d

from konrad.component import Component
from konrad.physics import vmr2relative_humidity
from konrad.utils import gaussian


__all__ = [
    "RelativeHumidityModel",
    "CacheFromAtmosphere",
    "VerticallyUniform",
    "Manabe67",
    "Cess76",
    "Romps14",
    "PolynomialCshapedRH",
    "PerturbProfile",
    "ProfileFromData",
]


class RelativeHumidityModel(Component, metaclass=abc.ABCMeta):
    def __call__(self, atmosphere, **kwargs):
        """Return the vertical distribution of relative humidity.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere: Atmosphere component.
            **kwargs: Arbitrary number of additional arguments,
                depending on the actual implementation.

        Returns:
            ndarray: Relative humidity profile.
        """
        ...


class CacheFromAtmosphere(RelativeHumidityModel):
    """Calculate and cache relative humidity from initial atmosphere."""

    def __init__(self):
        self._rh_profile = None

    def __call__(self, atmosphere, **kwargs):
        if self._rh_profile is None:
            self._rh_profile = vmr2relative_humidity(
                vmr=atmosphere["H2O"][-1],
                pressure=atmosphere["plev"],
                temperature=atmosphere["T"][-1],
            )
        return self._rh_profile


class VerticallyUniform(RelativeHumidityModel):
    """Fix the relative humidity to a single value throughout the atmosphere."""

    def __init__(self, rh_surface=0.8):
        """
        Parameters:
            rh_surface (float): Relative humidity at first
                pressure level (surface)
        """
        self.rh_surface = rh_surface
        self._rh_cache = None

    def __call__(self, atmosphere, **kwargs):
        if self._rh_cache is None:
            p = atmosphere["plev"]
            self._rh_cache = self.rh_surface * np.ones_like(p)

        return self._rh_cache


class Manabe67(RelativeHumidityModel):
    """Relative humidity model following Manabe and Wetherald (1967)."""

    def __init__(self, rh_surface=0.77):
        """Initialize a humidity model.

        Parameters:
            rh_surface (float): Relative humidity at the surface.
        """
        self.rh_surface = rh_surface

    def __call__(self, atmosphere, **kwargs):
        p = atmosphere["plev"]

        return self.rh_surface * (p / p[0] - 0.02) / (1 - 0.02)


class Cess76(RelativeHumidityModel):
    """Relative humidity model following Cess (1976).

    The relative humidity profile depends on the surface temperature.
    This results in moister atmospheres at warmer temperatures.
    """

    def __init__(self, rh_surface=0.8, T_surface=288):
        """Initialize a humidity model.

        Parameters:
            rh_surface (float): Relative humidity at the surface.
            T_surface (float): Surface temperature [K].
        """
        self.rh_surface = rh_surface
        self.T_surface = T_surface

    @property
    def omega(self):
        """Temperature dependent scaling factor for the RH profile."""
        return 1.0 - 0.03 * (self.T_surface - 288)

    def __call__(self, atmosphere, surface, **kwargs):
        p = atmosphere["plev"]
        self.T_surface = surface["temperature"][-1]

        return self.rh_surface * ((p / p[0] - 0.02) / (1 - 0.02)) ** self.omega


class Romps14(RelativeHumidityModel):
    """Relative humidity following an invariant RH-T relation."""

    def __init__(self):
        self._rh_func = None

    def __call__(self, atmosphere, **kwargs):
        if self._rh_func is None:
            self._rh_func = interp1d(
                # Values read from Fig. 6 in Romps (2014).
                x=np.array([300, 240, 200, 190, 188, 186]),
                y=np.array([0.8, 0.6, 0.7, 1.0, 0.5, 0.1]),
                kind="linear",
                fill_value="extrapolate",
            )

        return self._rh_func(atmosphere["T"][-1, :])


class PolynomialCshapedRH(RelativeHumidityModel):
    """Defines a C-shaped polynomial model, that depends on T in the upper troposphere.

    The RH increases linearly in the boundary layer from the surface.  Between
    the top of the boundary layer and the freezing level (T=273.15K), the rh is
    a quadratic function of p, defined by its values at these to points, and a
    zero derivative at the freezing level.  Above the freezing level, the rh is
    a quadratic function of T, defined by its values at the freezing level and
    at a chose upper-tropospheric T-value or at the cold point (see `top_peak_T`
    argument), and a zero derivative at the freezing level.

    """

    def __init__(
        self,
        top_peak_T=200.0,
        top_peak_rh=0.75,
        freezing_pt_rh=0.4,
        bl_top_p=940e2,
        bl_top_rh=0.85,
        surface_rh=0.75,
    ):
        """
        Parameters:
            top_peak_T (float): Temperature of the upper tropospheric peak. If None, coupled to the cold-point.
            top_peak_rh (float in [0;1]): value of relative humidity at the upper-tropospheric peak.
            freezing_pt_rh (float in [0;1]): value of relative humidity at the freezing point.
            bl_top_p (float): Pressure of the top of the boundary layer (bl) where the humidity peak is.
            bl_top_rh (float in [0;1]): value of the relative humidity at the top of the boundary layer.
            surface_rh (float in [0;1]): value of the relative humidity at the surface.
        """

        ## Convert percent to dimensionless
        if any(np.array([top_peak_rh, freezing_pt_rh, bl_top_rh, surface_rh]) > 1):
            raise ValueError(
                "Some RH values are given above 1, make sure RH is not in %."
                "If this was done on purpose, ignore this warning"
            )

        # Affect values to self
        self.top_peak_T = top_peak_T
        self.top_peak_rh = top_peak_rh
        self.freezing_pt_rh = freezing_pt_rh
        self.bl_top_p = bl_top_p
        self.bl_top_rh = bl_top_rh
        self.surface_rh = surface_rh

    def __call__(self, atmosphere, **kwargs):
        """
        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): The atmosphere component.

        Returns:
            ndarray: The relative humidity profile.
        """

        plev = atmosphere["plev"]
        T = atmosphere["T"][-1, :]

        ## Boundary layer
        bl_slope = (self.bl_top_rh - self.surface_rh) / (self.bl_top_p - 1000e2)

        def bl_func(p):
            return self.bl_top_rh + bl_slope * (p - self.bl_top_p)

        bl_rh = bl_func(plev[plev > self.bl_top_p])

        ## Between the top of the b.l. and the freezing point (fp)
        fp_p = atmosphere.get_triple_point_plev(interpolate=True)
        # Quadratic function of p going through both point with a zero slope at freezing level:
        def bottom_func(p):
            return (self.bl_top_rh - self.freezing_pt_rh) / (
                self.bl_top_p - fp_p
            ) ** 2 * (p - fp_p) ** 2 + self.freezing_pt_rh

        bottom_rh = bottom_func(plev[(plev <= self.bl_top_p) & (plev > fp_p)])

        ## Between the freezing point and the cold-point
        fp_T = 273.15
        if self.top_peak_T == None:
            top_peak_T = atmosphere["T"][-1, atmosphere.get_cold_point_index()]
        else:
            top_peak_T = self.top_peak_T
        # Quadratic function of T going through both point with a zero slope at freezing level:
        def top_func(T):
            return (self.top_peak_rh - self.freezing_pt_rh) / (
                top_peak_T - fp_T
            ) ** 2 * (T - fp_T) ** 2 + self.freezing_pt_rh

        top_rh = top_func(T[plev <= fp_p])

        return np.concatenate([bl_rh, bottom_rh, top_rh])


class PerturbProfile(RelativeHumidityModel):
    """Wrapper to add a perturbation to a Relative Humidity profile."""

    def __init__(
        self,
        base_profile=VerticallyUniform(),
        shape="square",
        center_plev=500e2,
        width=50e2,
        intensity=0.1,
        fixed_T=False,
    ):
        """
        Parameters:
            base_profile (konrad.relative_humidity model): initial profile on which we will add the perturbation.
            shape (str): name of the shape of the perturbation.
                Implemented : "square", "gaussian". For a Dirac use a square with width 0.
            center_plev (float): Pressure of the center of the square perturbation in [Pa].
            width (float): width of the perturbation in [Pa].
            intensity (float): Change in RH where the profile is perturbed, positive or negative.
            Fixed_T (boolean): If set to true, the temperature at center_plev at the first step is kept as the central
                point for the perturbation throughout the simulation, and the pressure at the center of the perturbation
                is no longer constant.
        """

        self._base_profile = base_profile
        self._shape = shape
        self.center_plev = center_plev
        self.width = width
        self.fixed_T = fixed_T
        self.center_T = None

        if intensity > 1:  # If intensity given in percents
            intensity /= 100
        self.intensity = float(intensity)

    def __call__(self, atmosphere, **kwargs):
        """
        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): The atmosphere component.

        Returns:
            ndarray: The relative humidity profile.
        """

        plev = atmosphere["plev"]
        T = atmosphere["T"][-1]

        if self.center_T == None:  # Initialize T at center_plev at the first step
            idx_center = np.abs(plev - self.center_plev).argmin()
            self.center_T = T[idx_center]

        if self.fixed_T:  # Compute center_plev to correspond to the fixed T
            T_ma = np.ma.masked_array(T, plev < atmosphere.pmin)
            idx_center = np.abs(T_ma - self.center_T).argmin()
            self.center_plev = plev[idx_center]
            self.width = konrad.utils.dp_from_dz()

        rh_profile = self._base_profile(atmosphere).copy()

        if self._shape == "square":
            idx_low = np.abs(plev - (self.center_plev + self.width / 2)).argmin()
            idx_high = np.abs(plev - (self.center_plev - self.width / 2)).argmin()
            if idx_low != idx_high:
                rh_profile[idx_low:idx_high] += self.intensity
            else:
                rh_profile[idx_low] += self.intensity

        if self._shape == "gaussian":
            G = gaussian(plev, self.center_plev, self.width / 2)  # Gaussian profile

            # Compute boundary of the perturbation
            p_low = self.center_plev + 1.5 * self.width
            idx_low = np.abs(plev - p_low).argmin()
            p_high = self.center_plev - 1.5 * self.width
            idx_high = np.abs(plev - p_high).argmin()
            if idx_low != idx_high:
                rh_profile[idx_low:idx_high] = (
                    rh_profile[idx_low:idx_high]
                    + G[idx_low:idx_high] / np.max(G) * self.intensity
                )
            else:
                rh_profile[idx_low] += self.intensity

        return rh_profile


class ProfileFromData(RelativeHumidityModel):
    """
    Defines a relative humidity from data.
    """

    def __init__(self, p_data, rh_data):
        """
        Parameters:
            p_data (np.ndarray): pressure coordinates corresponding to rh_data, in Pa
            rh_data (np.ndarray): the rh profile on p_data, in unit of RH
        """

        self._rh_func = interp1d(p_data, rh_data, fill_value="extrapolate")

    def __call__(self, atmosphere, **kwargs):
        """Return the vertical distribution of relative humidity.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere: Atmosphere component.
            **kwargs: Arbitrary number of additional arguments,
                depending on the actual implementation.

        Returns:
            ndarray: Relative humidity profile.
        """

        plev = atmosphere["plev"]
        return self._rh_func(plev)
