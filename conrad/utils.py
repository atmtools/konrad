# -*- coding: utf-8 -*-
"""Common utility functions.
"""
import os
import logging
from functools import wraps
from time import ctime

import numpy as np
import pandas as pd
import typhon
from netCDF4 import Dataset


__all__ = [
    'atmfield2pandas',
    'PsradSymlinks',
    'with_psrad_symlinks',
    'create_netcdf',
    'append_timestep_netcdf',
]

logger = logging.getLogger(__name__)


def atmfield2pandas(gf):
    """Convert a atm_field_compact to pandas DataFrame."""
    # List PSRAD variable names and corresponding ARTS species tags.
    psrad_keys = ['Z', 'T', 'Q', 'N2O', 'O3', 'CO', 'CH4']
    arts_keys = ['z', 'T', 'abs_species-H2O', 'abs_species-N2O',
                 'abs_species-O3', 'abs_species-CO', 'abs_species-CH4']

    # Store GriddedField fields in dict, matching PSRAD name is the key.
    data = {}
    for p, a in zip(psrad_keys, arts_keys):
        data[p] = typhon.arts.atm_fields_compact_get([a], gf).ravel()

    data['P'] = gf.grids[1]  # Store pressure grid as `P` variable.

    return pd.DataFrame({k: pd.Series(data[k], index=data['P'])
                         for k in data.keys()})


class PsradSymlinks():
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


def create_netcdf(filename, pressure, description='',
                  variable_description=None, clobber=True, zlib=True):
    """Create netCDF4 file for atmospheric soundings.

    This function initializes an empty netCDF4 file to store
    timeseries of atmospheric soundings. The file created has the
    the fixed dimension `pressure` and the unlimited dimension `time`.

    Parameters:
        filename (str): Path to the netCDF4.
        description (str): Data set description.
        pressure (ndarray): Pressure grid.
        variable_description (dict{tuple}): Description of variables to create.
            The dictionary keys are used as variable name. The dictionary value
            is a tuple with the varible long name and unit: `(long name, unit)`
        clobber (bool): If `True`, opening a file will clobber an 
            existing file.
        zlib (bool): Enable data compression.
    """
    # Open netCDF4 file in `write` mode. Non-existing files are created,
    # existing files are overwritten.
    with Dataset(filename, 'w', clobber=clobber, format='NETCDF4') as rootgrp:
        rootgrp.description = description
        rootgrp.history = 'Created ' + ctime()

        # Creating dimensions.
        rootgrp.createDimension('time', None)
        rootgrp.createDimension('pressure', np.size(pressure))

        # Creating variables.
        t = rootgrp.createVariable('time', float, ('time',), zlib=zlib)
        t.long_name = 'Time'
        t.units = 'hours since 0001-01-01 00:00:00.0'
        t.calendar = 'gregorian'

        p = rootgrp.createVariable('P', float, ('pressure',), zlib=zlib)
        if hasattr(pressure, 'values'):
            p[:] = pressure.values
        else:
            p[:] = pressure
        p.long_name = 'Pressure'
        p.units = 'Pa'

        # Create each variable defined in ``variable_description``.
        if variable_description is not None:
            for var, (name, unit) in variable_description.items():
                v = rootgrp.createVariable(
                    var, float, ('time', 'pressure',), zlib=zlib)
                v.long_name = name
                v.units = unit
                logging.debug(
                    'Added variable "{}" to "{}".'.format(var, filename))

        logging.info('Created "{}".'.format(filename))


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
        logging.debug( 'Append timestep to "{}".'.format(filename))
        t = nc.dimensions['time'].size  # get index to store data.
        nc.variables['time'][t] = timestamp  # append timestamp.

        # Append data for each variable in ``data`` that has the
        # dimensions `time` and `pressure`.
        for var in data:
            if nc[var].dimensions == ('time', 'pressure'):
                if hasattr(data[var], 'values'):
                    nc.variables[var][t, :] = data[var].values
                else:
                    nc.variables[var][t, :] = data[var]
