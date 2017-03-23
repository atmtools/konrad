# -*- coding: utf-8 -*-
"""Core functionality for radiative-convective models.
"""
from typhon import atmosphere


__all__ = [
    'adjust_vmr',
]


def adjust_vmr(sounding, T_new):
    rh = atmosphere.relative_humidity(
            sounding['Q'], sounding['P'], sounding['T'])

    return atmosphere.vmr(rh, sounding['P'], T_new)
