# -*- coding: utf-8 -*-
"""Common utility functions.
"""
import logging

from netCDF4 import Dataset


__all__ = [
    'append_timestep_netcdf',
]

logger = logging.getLogger(__name__)


def append_timestep_netcdf(filename, data, timestamp):
    """Append a timestep to an existing variable in a netCDF4 file.

    The variable has to be existing in the netCDF4 file
    as the values are **appended**.

    Parameters:
        filename (str): Path to the netCDF4.
        data (dict{ndarray}): Dictionary containing the data to append.
            The key is the variable name and the value is an `ndarray`
            matching the variable dimensions. e.g.:
                data = {'T': np.array(290, 295, 300)}
        timestamp (float): Timestamp of values appended.
    """
    # Open netCDF4 file in `append` mode.
    with Dataset(filename, 'a') as nc:
        logging.debug('Append timestep to "{}".'.format(filename))
        t = nc.dimensions['time'].size  # get index to store data.
        nc.variables['time'][t] = timestamp  # append timestamp.

        # Append data for each variable in ``data`` that has the
        # dimensions `time` and `pressure`.
        for var in data:
            if nc[var].dimensions == ('time', 'plev'):
                if hasattr(data[var], 'values'):
                    nc.variables[var][t, :] = data[var].values
                else:
                    nc.variables[var][t, :] = data[var]
