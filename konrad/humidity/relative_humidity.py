"""Different models to describe the vertical relative humidity distribution."""
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d


class HeightConstant:
    def __init__(self, rh_surface=0.62):
        self.rh_surface = rh_surface
        self._rh_cache = None

    def __call__(self, atmosphere, **kwargs):
        """

        Parameters:
            atmosphere (``konrad.atmosphere.Atmosphere``):
                Atmosphere component.
            **kwargs: Additional components may be passed as keyword arguments.
                Depending on the used humidity model their are used or ignored.
        """
        if self._rh_cache is None:
            p = atmosphere['plev']
            self._rh_cache = self.rh_surface * np.ones_like(p)

        return self._rh_cache


class ConstantFreezingLevel:
    """Constant rel. humidity up to the freezing level and then decreasing."""
    def __init__(self, rh_surface=0.77):
        self.rh_surface = rh_surface

    def __call__(self, atmosphere, **kwargs):
        plev = atmosphere['plev']
        rh_profile = self.rh_surface * np.ones_like(plev)

        fl = atmosphere.get_triple_point_index()
        rh_profile[fl:] = (
            self.rh_surface * (plev[fl:] / plev[fl])**(1/4)
        )

        return rh_profile


class FixedUTH:
    """Idealised model of a fixed C-shaped relative humidity distribution."""
    def __init__(self, rh_surface=0.8, uth=0.8, uth_plev=170e2, uth_offset=0):
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

    def get_relative_humidity_profile(self, atmosphere):
        p = atmosphere['plev']

        # Exponential decay from the surface value throughout the atmosphere.
        rh = self.rh_surface * (p / p[0]) ** 1.15

        # Add skew-normal distribution.
        uth = self.uth * norm.pdf(
            x=np.log(p),
            loc=np.log(self.uth_plev + self.uth_offset),
            scale=0.4,
        )

        return np.maximum(rh, uth)

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


class Manabe67:
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


class Cess76:
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


class Romps14:
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
