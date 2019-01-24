"""This module contains functions related to physics."""
import typhon.physics as typ


def saturation_pressure(temperature, Tmin=130, Tmax=340, **kwargs):
    """Calculate equilibrium vapor pressure of water.

    The equilibrium vapor pressure is defined with respect to saturation
    over ice below -23°C and with respect to saturation over water above 0°C.
    In between an interpolation is applied (defaults to ``quadratic``).

    Parameters:
        temperature (float or ndarray): Temperature [K].
        Tmin (float): Lower bound of temperature interpolation [K].
        Tmax (float): Upper bound of temperature interpolation [K].
        **kwargs: All remaining keyword arguments are passed to
            :func:`typhon.physics.e_eq_mixed_mk`.

    Returns:
        float or ndarray: Equilibrium vapor pressure [Pa].

    References:
        Murphy, D. M. and Koop, T. (2005): Review of the vapour pressures of
        ice and supercooled water for atmospheric applications,
        Quarterly Journal of the Royal Meteorological Society 131(608):
        1539–1565. doi:10.1256/qj.04.94
        """
    return typ.e_eq_mixed_mk(temperature, Tmin=Tmin, Tmax=Tmax, **kwargs)


def relative_humidity2vmr(relative_humidity, pressure, temperature):
    r"""Convert relative humidity into water vapor VMR.

    .. math::
        VMR = \frac{RH \cdot e_s(T)}{p}

    Parameters:
        relative_humidity (float or ndarray): Relative humidity.
        pressure (float or ndarray): Pressue [Pa].
        temperature(float or ndarray): Temperature [K].

    Returns:
        float or ndarray: Water vapor volume mixing ratio [dimensionless].
    """
    return typ.relative_humidity2vmr(
        RH=relative_humidity,
        p=pressure,
        T=temperature,
        e_eq=saturation_pressure,
    )


def vmr2relative_humidity(vmr, pressure, temperature):
    r"""Convert water vapor VMR into relative humidity.

    .. math::
        RH = \frac{VMR \cdot p}{e_s(T)}

    Parameters:
        vmr (float or ndarray): Water vapor volume mixing ratio.
        pressure (float or ndarray): Pressure [Pa].
        temperature (float or ndarray): Temperature [K].

    Returns:
        float or ndarray: Relative humidity [dimensionless].
    """
    return typ.vmr2relative_humidity(
        vmr=vmr,
        p=pressure,
        T=temperature,
        e_eq=saturation_pressure,
    )
