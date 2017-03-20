# -*- coding: utf-8 -*-
"""Core functionality for radiative-convective models.
"""
from typhon import atmosphere


__all__ = [
    'adjust_temperature',
    'adjust_vmr',
]


def adjust_temperature(c, convective_adjustment=False):
    if convective_adjustment:
        logging.warn('Convective adjustment is not implemented yet.')

    net_rate = (c.rad_lw['lw_htngrt'] + c.rad_sw['sw_htngrt'])

    return c.sounding['T'] + net_rate


def adjust_vmr(sounding, T_new):
    rh = atmosphere.relative_humidity(
            sounding['Q'], sounding['P'], sounding['T'])

    return atmosphere.vmr(rh, sounding['P'], T_new)
