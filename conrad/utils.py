# -*- coding: utf-8 -*-
"""Common utility functions.
"""
import os
import logging
from functools import wraps

import pandas as pd
import typhon


__all__ = [
    'atmfield2pandas',
    'PsradSymlinks',
    'with_psrad_symlinks',
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

    # Unit conversion.
    data['P'] = gf.grids[1] / 100
    data['Q'] *= 1000
    data['O3'] *= 1e+06
    data['N2O'] *= 1e+06
    data['CO'] *= 1e+06
    data['CH4'] *= 1e+06

    return pd.DataFrame({k: pd.Series(data[k], index=data['P'])
                         for k in data.keys()})


class PsradSymlinks():
    def __init__(self):
        try:
            self._psrad_path = os.environ['PSRAD_PATH']
        except KeyError:
            logger.exception('Path to PSRAD directory not set.')

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
