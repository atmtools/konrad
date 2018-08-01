# -*- coding: utf-8 -*-
"""Common utility functions.
"""
import copy
import logging

import numpy as np
import typhon as ty
from netCDF4 import Dataset

from konrad import constants


__all__ = [
    'append_description',
    'append_timestep_netcdf',
    'return_if_type',
    'phlev_from_plev',
    'refined_pgrid',
    'get_pressure_grids',
    'ozonesquash',
    'ozone_profile_rcemip',
]

logger = logging.getLogger(__name__)


def append_description(dataset, description=None):
    """Append variable attributes to a given dataset.

    Parameters:
          dataset (xarray.Dataset): Dataset including variables to describe.
          description (dict): Dictionary containing variable descriptions.
            The keys are the variable keys used in the Dataset.
            The values are dictionaries themselves containing attributes
            and their names as keys, e.g.:
                desc = {'T': {'units': 'K', 'standard_name': 'temperature'}}
    """
    if description is None:
        description = constants.variable_description

    for key in dataset.variables:
        if key in description:
            dataset[key].attrs = constants.variable_description[key]


def append_timestep_netcdf(filename, data, timestamp):
    """Append a timestep to an existing netCDF4 file.

    Notes:
        The variables to append to have to exist in the netCDF4 file!

    Parameters:
        filename (str): Path to the netCDF4.
        data (dict{ndarray}): Dict-like object containing the data arrays.
            The key is the variable name and the value is an ``ndarray``, a
            ``pandas.Series`` or an ``xarray.DataArray`` e.g.:
                >>> data = {'T': np.array([290, 295, 300])}

        timestamp (float): Timestamp of values appended.
    """
    # Open netCDF4 file in `append` mode.
    with Dataset(filename, 'a') as nc:
        logging.debug('Append timestep to "{}".'.format(filename))
        t = nc.dimensions['time'].size  # get index to store data.
        nc.variables['time'][t] = timestamp  # append timestamp.

        # Append values for each data variable in ``data``.
        for var in data.data_vars:
            # Append variable if it has a `time` dimension.
            if 'time' in nc[var].dimensions:
                # TODO: Find a cleaner way to handle different data dimensions.
                if 'plev' in nc[var].dimensions:
                    if hasattr(data[var], 'values'):
                        nc.variables[var][t, :] = data[var].values
                    else:
                        nc.variables[var][t, :] = data[var]
                else:
                    if hasattr(data[var], 'values'):
                        nc.variables[var][t] = data[var].values
                    else:
                        nc.variables[var][t] = data[var]


def return_if_type(variable, variablename, expect, default):
    """Return a variable if it matches an expected type.

    Parameters:
          variable: Variable to check.
          variablename (str): Variable name for error message.
          expect (type): Expected variable type.
          default: Default value, if varibale is ``None``.

    Raises:
          TypeError: If variable does not match expected type.
    """
    if variable is None:
        # use a surface with heat capacity as default.
        variable = default
    elif not isinstance(variable, expect):
        raise TypeError(
            'Argument `{name}` has to be of type `{type}`.'.format(
                name=variablename, type=expect.__name__)
        )

    return variable


def phlev_from_plev(fulllevels):
    """Returns the linear interpolated halflevels for given array.

    Parameters:
        fulllevels (ndarray): Pressure at fullevels.

    Returns:
        ndarray: Coordinates at halflevel.

    """
    inter = (fulllevels[1:] + fulllevels[:-1]) / 2
    bottom = fulllevels[0] - 0.5 * (fulllevels[1] - fulllevels[0])
    top = 0
    return np.hstack((bottom, inter, top))


def refined_pgrid(start, stop, num=200, shift=0.5, fixpoint=0.):
    """Create a pressure grid with adjustable distribution in logspace.

    Notes:
          Wrapper for ``typhon.math.squeezable_logspace``.

    Parameters:
        start (float): The starting value of the sequence.
        stop (float): The end value of the sequence.
        num (int): Number of sample to generate (Default is 50).
        shift (float): Factor with which the first stepwidth is
            squeezed in logspace. Has to be between  ``(0, 2)`.
            Values smaller than one compress the gridpoints,
            while values greater than 1 strecht the spacing.
            The default is ``0.5`` (bottom heavy.)
        fixpoint (float): Relative fixpoint for squeezing the grid.
            Has to be between ``[0, 1]``. The  default is ``0`` (bottom).

    Returns:
        ndarray: Pressure grid.
    """
    grid = ty.math.squeezable_logspace(
        start=start, stop=stop, num=num, squeeze=shift, fixpoint=fixpoint
    )

    return grid


def get_pressure_grids(start=1000e2, stop=1, num=200, squeeze=0.5):
    """Create matching pressure levels and half-levels.

    Parameters:
        start (float): Pressure of the lowest half-level (surface) [Pa].
        stop (float): Pressure of the highest half-level (TOA) [Pa].
        num (int): Number of **full** pressure levels.
        squeeze (float): Factor with which the first step width is
            squeezed in logspace. Has to be between ``(0, 2)``.
            Values smaller than one compress the half-levels,
            while values greater than 1 stretch the spacing.
            The default is ``0.5`` (bottom heavy.)

    Returns:
        ndarray, ndarray: Full-level pressure, half-level pressure [Pa].
    """
    phlev = ty.math.squeezable_logspace(start, stop, num + 1, squeeze=squeeze)
    plev = np.exp(0.5 * (np.log(phlev[1:]) + np.log(phlev[:-1])))

    return plev, phlev


def ozonesquash(o3, z, squash):
    """
    Squashes the ozone profile upwards or stretches it downwards, with no
        change to the shape of the profile above the ozone concentration maximum
    Parameters:
        o3 (ndarray): initial ozone profile
        z (ndarray): corresponding height values
        squash: float, with 1 being no squash,
            numbers < 1 squashing the profile towards the maximum,
            numbers > 1, stretching the profile downwards
    Returns:
        ndarray: new ozone profile
    """
    i_max_o3 = np.argmax(o3)

    sqz = (z - z[i_max_o3])*squash + z[i_max_o3]
    new_o3 = copy.copy(o3)
    new_o3[:i_max_o3] = np.interp(z[:i_max_o3], sqz, o3)
    return new_o3


def ozone_profile_rcemip(plev, g1=3.6478, g2=0.83209, g3=11.3515):
    """Compute the ozone volumetric mixing ratio from pressure.

    .. math::
        O_3 = g_1 \cdot p^{g_2} e^\frac{-p}{g_3}

    Parameters:
        plev (ndarray): Atmospheric pressure [Pa].
        g1, g2, g3 (float): Fitting parameters for gamma distribution
            according to Wing et al. (2017).

    Returns:
          ndarray: Ozone profile [VMR].

    Reference:
        Wing et al., 2017, Radiative-Convective Equilibrium Model
        Intercomparison Project

    """
    p = plev / 100
    return g1 * p**g2 * np.exp(-p / g3) * 1e-6
