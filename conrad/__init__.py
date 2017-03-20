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
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    style='{',
    format='{asctime} {levelname}:{name}:{message}',
    )
