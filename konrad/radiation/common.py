import numpy as np

from konrad import constants


def fluxes2heating(net_fluxes, pressure, cp=None, method='diff'):
    r"""Calculate radiative heating from net fluxes

    .. math::
        Q_\mathrm{r} = \frac{F}{c_p} \frac{\mathrm{d}F}{\mathrm{d}p}

    Parameters:
        net_fluxes (ndarray): Net radiative flux.
        pressure (ndarray): Pressure level.
        cp (float or ndarray): Specific heat capacity. If ``None`` use
            specific heat capacity of dry air.
        method (str): Method used to derive the radiative fluxes
            ("diff" or "gradient").

    Returns:
        ndarray: Radiative heating [K/day].
    """
    g = constants.earth_standard_gravity
    if cp is None:
        cp = constants.isobaric_mass_heat_capacity_dry_air

    if method == 'diff':
        dfdp = np.diff(net_fluxes) / np.diff(pressure)
    elif method == 'gradient':
        dfdp = np.gradient(net_fluxes, pressure)
    else:
        raise ValueError(f'Method has to be "diff" or "gradient".')

    heating = g / cp * dfdp

    return heating * constants.seconds_in_a_day  # K/s -> K/day
