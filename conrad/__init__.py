# -*- coding: utf-8 -*-
"""Radiative-convective equilibrium model.
"""
import logging

from . import plots
from . import utils
from .conrad import ConRad


__all__ = [
    'ConRad',
    'plots',
    'utils',
]

__version__ = '0.1'

logging.basicConfig(
    # filename='log.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
