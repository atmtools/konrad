# -*- coding: utf-8 -*-
"""This package aims to provide a framework to run radiative-convective
equilibrium (RCE) simulations.

The package is written in a strict object - oriented way. The RCE model is
subdivided into self-contained submodels(atmosphere, surface, radiation) that
can be implemented independently from each other. This allows to exchange parts
of the model setup separately to investigate certain effects. The different
submodels need to fulfill requirements to make interaction possible. These
requirements are enforced by the use of abstract base classes.
"""
import logging
from os.path import (join, dirname)

__version__ = open(join(dirname(__file__), 'VERSION')).read().strip()

from . import aerosol
from . import atmosphere
from . import cloud
from . import component
from . import constants
from . import convection
from . import humidity
from . import lapserate
from . import netcdf
from . import ozone
from . import physics
from . import plots
from . import radiation
from . import surface
from . import upwelling
from . import utils
from .core import RCE


# Basic configuration for all loggers used within konrad.
# NOTE: The process name is included for more verbose logs in multiprocessing.
logging.basicConfig(
    # filename='log.txt',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    style='{',  # Allows to use format string syntax in the next line.
    format='{asctime} {processName}:{levelname}:{name}:{message}',
    )
