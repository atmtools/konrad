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
from . import convection
from . import constants
from . import humidity
from . import lapserate
from . import plots
from . import radiation
from . import surface
from . import upwelling
from . import utils
from .core import RCE


__all__ = [
    'RCE',
    'atmosphere',
    'constants',
    'convection',
    'humidity',
    'lapserate',
    'plots',
    'radiation',
    'surface',
    'upwelling',
    'utils',
]

# Basic configuration for all loggers used within konrad.
# NOTE: The process name is included for more verbose logs in multiprocessing.
logging.basicConfig(
    # filename='log.txt',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    style='{',  # Allows to use format string syntax in the next line.
    format='{asctime} {processName}:{levelname}:{name}:{message}',
    )
