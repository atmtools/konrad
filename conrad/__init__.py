# -*- coding: utf-8 -*-
"""This package aims to provide a framework to run radiative-convective
equilibirum (RCE) simulations.

The package is written in a strict object - oriented way. The RCE model is
subdivided into self-contained submodels(atmosphere, surface, radiation) that
can be implemented independently from each other. This allows to exchange parts
of the model setup seperately to investigate certain effects. The different
submodels need to fulfill requirements to make interaction possible. These
requirements are enforced by the use of abstract base classes.
"""
import logging

from . import atmosphere
from . import constants
from . import plots
from . import radiation
from . import surface
from . import utils
from .conrad import RCE


__all__ = [
    'RCE',
    'atmosphere',
    'constants',
    'plots',
    'radiation',
    'surface',
    'utils',
]

__version__ = '0.2'

logging.basicConfig(
    # filename='log.txt',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S',
    style='{',
    format='{asctime} {levelname}:{name}:{message}',
    )
