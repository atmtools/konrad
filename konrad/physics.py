"""This module contains functions related to physics."""
from numbers import Number

import numpy as np
import typhon.physics as typ

from konrad import constants


# TODO: Replace with typhon version after next release (>0.7.0)
def saturation_pressure(temperature):
    r"""Return equilibrium pressure of water with respect to the mixed-phase.

    The equilibrium pressure over water is taken for temperatures above the
    triple point :math:`T_t` the value over ice is taken for temperatures
    below :math:`T_t–23\,\mathrm{K}`.  For intermediate temperatures the
    equilibrium pressure is computed as a combination
    of the values over water and ice according to the IFS documentation:

    .. math::
        e_\mathrm{s} = \begin{cases}
            T > T_t, & e_\mathrm{liq} \\
            T < T_t - 23\,\mathrm{K}, & e_\mathrm{ice} \\
            else, & e_\mathrm{ice}
                + (e_\mathrm{liq} - e_\mathrm{ice})
                \cdot \left(\frac{T - T_t - 23}{23}\right)^2
        \end{cases}

    References:
        IFS Documentation – Cy45r1,
        Operational implementation 5 June 2018,
        Part IV: Physical Processes, Chapter 12, Eq. 12.13,
        https://www.ecmwf.int/node/18714

    Parameters:
        temperature (float or ndarray): Temperature [K].

    See also:
        :func:`~typhon.physics.e_eq_ice_mk`
            Equilibrium pressure of water over ice.
        :func:`~typhon.physics.e_eq_water_mk`
            Equilibrium pressure of water over liquid water.

    Returns:
        float or ndarray: Equilibrium pressure [Pa].
    """
    # Keep track of input type to match the return type.
    is_float_input = isinstance(temperature, Number)
    if is_float_input:
        # Convert float input to ndarray to allow indexing.
        temperature = np.asarray([temperature])

    e_eq_water = typ.e_eq_water_mk(temperature)
    e_eq_ice = typ.e_eq_ice_mk(temperature)

    is_water = temperature > constants.triple_point_water

    is_ice = temperature < (constants.triple_point_water - 23.)

    e_eq = (e_eq_ice + (e_eq_water - e_eq_ice)
            * ((temperature - constants.triple_point_water + 23) / 23)**2
            )
    e_eq[is_ice] = e_eq_ice[is_ice]
    e_eq[is_water] = e_eq_water[is_water]

    return float(e_eq) if is_float_input else e_eq


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
