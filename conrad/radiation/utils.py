# -*- coding: utf-8 -*-
"""Utility functions related to radiation models.
"""
import os
import logging
from functools import wraps


__all__ = [
    'PsradSymlinks',
    'with_psrad_symlinks',
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
        ['ECHAM6_CldOptProps.nc',
         'rrtmg_lw.nc',
         'rrtmg_sw.nc',
         'libpsrad.so.1']

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
