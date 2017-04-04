# -*- coding: utf-8 -*-
"""Common utility functions.
"""
import os
import logging
from functools import wraps

from netCDF4 import Dataset


__all__ = [
    'PsradSymlinks',
    'with_psrad_symlinks',
    'append_timestep_netcdf',
]

logger = logging.getLogger(__name__)


class PsradSymlinks():
    """Defines a with-block to ensure that all files needed to run PSRAD are
    symlinked.

    Examples:
        >>> print(os.listdir())
        []
        >>> with PsradSymlinks():
        ...     print(os.listdir())
        ['ECHAM6_CldOptProps.nc', 'rrtmg_lw.nc', 'rrtmg_sw.nc', 'libpsrad.so.1']

    """
    def __init__(self):
        try:
            self._psrad_path = os.environ['PSRAD_PATH']
        except KeyError:
            logger.exception('Path to PSRAD directory not set.')
            raise

        self._psrad_files = [
            'ECHAM6_CldOptProps.nc',
            'rrtmg_lw.nc',
            'rrtmg_sw.nc',
            'libpsrad.so.1',
            ]
        self._created_files = []

    def __enter__(self):
        for f in self._psrad_files:
            if not os.path.isfile(f):
                os.symlink(os.path.join(self._psrad_path, f), f)
                self._created_files.append(f)
                logger.debug("Create symlink %s", f)

    def __exit__(self, *args):
        for f in self._created_files:
            os.remove(f)


def with_psrad_symlinks(func):
    """Wrapper for all functions that import the psrad module.

    The decorator asures that ´libpsrad.so.1´ and the requied *.nc files are
    symlinked in the current working directory. This allows a more flexible
    usage of the psrad module.
    """
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        with PsradSymlinks():
            return func(*args, **kwargs)
    return func_wrapper


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
