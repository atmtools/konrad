#!/usr/bin/env python3
"""Extract last timestep of given netCDF files and store them as ARTS XMl file.

Usage:
    $ python netcdf2artsxml.py rce1.nc rce2.nc batch_atm_fields_compact.xml
"""
import sys

from konrad.atmosphere import Atmosphere
from typhon.arts import xml


xml.save([Atmosphere.from_netcdf(ncfile).to_atm_fields_compact()
          for ncfile in sys.argv[1:-1]], sys.argv[-1])
