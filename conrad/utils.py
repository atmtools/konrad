# -*- coding: utf-8 -*-
"""Common utility functions.
"""
import logging

import numpy as np

from netCDF4 import Dataset


__all__ = [
    'append_timestep_netcdf',
    'create_relative_humidity_profile',
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
            if nc[var].dimensions == ('time', 'plev'):
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
