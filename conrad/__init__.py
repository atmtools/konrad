# -*- coding: utf-8 -*-
"""Radiative-convective equilibrium model.
"""
import logging

from . import atmosphere
from . import plots
from . import radiation
from . import surface
from . import utils
from .conrad import ConRad


__all__ = [
    'ConRad',
    'atmosphere',
    'plots',
    'surface',
    'utils',
]

__version__ = '0.1'

logging.basicConfig(
    # filename='log.txt',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S',
    style='{',
    format='{asctime} {levelname}:{name}:{message}',
    )
