"""Different models to describe the vertical relative humidity distribution."""
import abc

import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

from konrad.component import Component


class RelativeHumidityModel(Component, metaclass=abc.ABCMeta):
    def __call__(self, atmosphere, **kwargs):
        """Return the vertical distirbution of relative humidity.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere: Atmosphere component.
            **kwargs: Arbitrary number of additional arguments,
                depending on the actual implementation.

        Returns:
            ndarray: Relative humidity profile.
        """
        ...


class HeightConstant(RelativeHumidityModel):
    """Constant relative humidity profile throughout the whole troposphere."""
    def __init__(self, rh_surface=0.62):
        self.rh_surface = rh_surface
        self._rh_cache = None

    def __call__(self, atmosphere, **kwargs):
        if self._rh_cache is None:
            p = atmosphere['plev']
            self._rh_cache = self.rh_surface * np.ones_like(p)

        return self._rh_cache


class VerticallyUniform(RelativeHumidityModel):
    def __init__(self, rh_surface=0.5, rh_tropopause=0.3):
        self.rh_surface = rh_surface
        self.rh_tropopause = rh_tropopause
        self.convective_top = 300e2
        self.cold_point = 100e2

    def __call__(self, atmosphere, convection, **kwargs):
        p = atmosphere['plev']
        self.convective_top = convection.get('convective_top_plev')[0]
        self.cold_point = atmosphere.get_cold_point_plev()

        rh = (
            (self.rh_tropopause - self.rh_surface)
            / (self.cold_point - self.convective_top)
            * (p - self.convective_top) + self.rh_surface
        )
        rh[p > self.convective_top] = self.rh_surface

        return rh


class ConstantFreezingLevel(RelativeHumidityModel):
    """Constant rel. humidity up to the freezing level and then decreasing."""
    def __init__(self, rh_surface=0.77):
        self.rh_surface = rh_surface

    def __call__(self, atmosphere, **kwargs):
        plev = atmosphere['plev']
        rh_profile = self.rh_surface * np.ones_like(plev)

        fl = atmosphere.get_triple_point_index()
        rh_profile[fl:] = (
            self.rh_surface * (plev[fl:] / plev[fl])**1.3
        )

        return rh_profile


class FixedUTH(RelativeHumidityModel):
    """Idealised model of a fixed C-shaped relative humidity distribution."""
    def __init__(self, rh_surface=0.77, uth=0.75, uth_plev=170e2,
                 uth_offset=0):
        """Couple the upper-tropospheric humidity peak to the convective top.

        Parameters:
            rh_surface (float): Relative humidity at first
                pressure level (surface).
            uth (float): Relative humidity at the upper-tropospheric peak.
            uth_plev (float): Pressure level of second humidity maximum [Pa].
            uth_offset (float): Offset between UTH peak and convective top.
        """
        self.rh_surface = rh_surface
        self.uth = uth
        self.uth_plev = uth_plev
        self.uth_offset = uth_offset

        self._rh_base_profile = None

    def get_relative_humidity_profile(self, atmosphere):
        p = atmosphere['plev']

        # Use Manabe (1967) relative humidity model as base/background.
        if self._rh_base_profile is None:
            manabe_model = Manabe67(rh_surface=self.rh_surface)
            self._rh_base_profile = manabe_model(atmosphere)

        # Add skew-normal distribution.
        uth = self.uth * norm.pdf(
            x=np.log(p),
            loc=np.log(self.uth_plev + self.uth_offset),
            scale=0.4,
        )

        return np.maximum(self._rh_base_profile, uth)

    def __call__(self, atmosphere, **kwargs):
        return self.get_relative_humidity_profile(atmosphere)


class CoupledUTH(FixedUTH):
    """Idealised model of a coupled C-shaped relative humidity distribution.

    This relative humidity works in the same way as ``FixedUTH`` but the
    ``uth_plev`` is updated automatically depending on the convective top.
    """
    def __call__(self, atmosphere, convection, **kwargs):
        self.uth_plev = convection.get('convective_top_plev')[0]

        return self.get_relative_humidity_profile(atmosphere)


class CshapeConstant(RelativeHumidityModel):
    """Idealized model of a C-shaped RH profile using a quadratic equation."""
    def __init__(self, uth_plev=200e2, rh_min=0.3, uth=0.8):
        self.uth_plev = uth_plev
        self.rh_min = rh_min
        self.uth = uth
        self.rh_surface = uth

    def __call__(self, atmosphere, convection, **kwargs):
        self.uth_plev = convection.get('convective_top_plev')[0]

        x = np.log10(atmosphere['plev'])
        xmin = np.log10(self.uth_plev)
        xmax = x[0]

        a = (self.uth - self.rh_min) * 4 / (xmin - xmax)**2
        b = (xmin + xmax) / 2
        c = self.rh_min

        return np.clip(a * (x - b)**2 + c, a_min=0, a_max=1)


class CshapeDecrease(RelativeHumidityModel):
    """Idealized model of a C-shaped RH profile using a quadratic equation."""
    def __init__(self, uth_plev=200e2, rh_min=0.3, uth=0.8):
        self.uth_plev = uth_plev
        self.rh_min = rh_min
        self.uth = uth
        self.rh_surface = uth

    def __call__(self, atmosphere, convection, **kwargs):
        self.uth_plev = convection.get('convective_top_plev')[0]

        x = np.log10(atmosphere['plev'])
        xmin = np.log10(self.uth_plev)
        xmax = x[0]

        a = (self.uth - self.rh_min) * 4 / (xmin - xmax)**2
        b = (xmin + xmax) / 2
        c = self.rh_min

        rh = np.clip(a * (x - b)**2 + c, a_min=0, a_max=1)

        rh[x < xmin] *= (10**x / 10**xmin)[x < xmin]

        return rh


class Manabe67(RelativeHumidityModel):
    """Relative humidity model following Manabe and Wetherald (1967)."""
    def __init__(self, rh_surface=0.77):
        """Initialize a humidity model.

        Parameters:
            rh_surface (float): Relative humidity at the surface.
        """
        self.rh_surface = rh_surface

    def __call__(self, atmosphere, **kwargs):
        p = atmosphere['plev']

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
        p = atmosphere['plev']
        self.T_surface = surface['temperature'][-1]

        return self.rh_surface * ((p / p[0] - 0.02) / (1 - 0.02))**self.omega


class Romps14(RelativeHumidityModel):
    """Return relative humidity according to an invariant RH-T relation."""
    def __call__(self, atmosphere, **kwargs):
        if self._rh_func is None:
            self._rh_func = interp1d(
                # Values read from Fig. 6 in Romps (2014).
                x=np.array([300, 240, 200, 190, 188, 186]),
                y=np.array([0.8, 0.6, 0.7, 1.0, 0.5, 0.1]),
                kind='linear',
                fill_value='extrapolate',
            )

        return self._rh_func(atmosphere['T'][-1, :])
