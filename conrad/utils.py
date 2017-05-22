# -*- coding: utf-8 -*-
"""Common utility functions.
"""
import logging

import numpy as np

from netCDF4 import Dataset


__all__ = [
    'append_timestep_netcdf',
    'create_relative_humidity_profile',
    'ensure_decrease',
    'calculate_halflevels',
]

logger = logging.getLogger(__name__)


def append_timestep_netcdf(filename, data, timestamp):
    """Append a timestep to an existing netCDF4 file.

    Notes:
        The variables to append to have to exist in the netCDF4 file!

    Parameters:
        filename (str): Path to the netCDF4.
        data (dict{ndarray}): Dict-like object containing the data arrays.
            The key is the variable name and the value is an ``ndarray``, a
            ``pandas.Series`` or an ``xarray.DataArray`` e.g.:
                >>> data = {'T': np.array(290, 295, 300)}

        timestamp (float): Timestamp of values appended.
    """
    # Open netCDF4 file in `append` mode.
    with Dataset(filename, 'a') as nc:
        logging.debug('Append timestep to "{}".'.format(filename))
        t = nc.dimensions['time'].size  # get index to store data.
        nc.variables['time'][t] = timestamp  # append timestamp.

        # Append data for each variable in ``data`` that has the
        # dimensions ``time`` and ``plev``.
        for var in data:
            # Append variable if it has a `time` dimension and is no
            # dimension itself.
            if 'time' in nc[var].dimensions and var not in nc.dimensions:
                if hasattr(data[var], 'values'):
                    nc.variables[var][t, :] = data[var].values
                else:
                    nc.variables[var][t, :] = data[var]


def create_relative_humidity_profile(p, RH_s=0.75):
    """Create an exponential relative humidity profile.

    Parameters:
        p (ndarray): Pressure.
        RH_s (float): Relative humidity at first pressure level.

    Returns:
        ndarray: Relative humidtiy."""
    rh = RH_s / (np.exp(1) - 1) * (np.exp(p / p[0]) - 1)
    return np.round(rh, decimals=4)


def ensure_decrease(array):
    """Ensure that a given array is decreasing.

    Parameters:
        array (ndarray): Input array.

    Returns:
        ndarray: Monotonously decreasing array.
    """
    for i in range(1, np.size(array)):
        if array[i] > array[i-1]:
            array[i] = array[i-1]
    return array


def calculate_halflevels(level):
    """Returns the linear inteprolated halflevels for given array.

    Parameters:
        level (ndarray): Data array.

    Returns:
        ndarray: Coordinates at halflevel.

    Examples:
        >>> interpolate_halflevels([0, 1, 2, 4])
        array([ 0.5,  1.5,  3. ])
    """
    inter = (level[1:] + level[:-1]) / 2
    bottom = level[0] - (level[1] - level[0])
    top = level[-1] / 2
    return np.hstack((bottom, inter, top))