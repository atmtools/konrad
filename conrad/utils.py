# -*- coding: utf-8 -*-
"""Common utility functions.
"""
from glob import glob
import pandas as pd
import typhon
from subprocess import call


__all__ = [
    'atmfield2pandas',
]


def atmfield2pandas(gf):
    """Convert a atm_field_compact to pandas DataFrame."""
    # List PSRAD variable names and corresponding ARTS species tags.
    psrad_keys = ['Z', 'T', 'Q', 'N2O', 'O3', 'CO', 'CH4']
    arts_keys = ['z', 'T', 'abs_species-H2O', 'abs_species-N2O',
                 'abs_species-O3', 'abs_species-CO', 'abs_species-CH4']

    # Store GriddedField fields in dict, matching PSRAD name is the key.
    data = {}
    for p, a in zip(psrad_keys, arts_keys):
        data[p] = typhon.arts.atm_fields_compact_get([a], gf).ravel()

    # Unit conversion.
    data['P'] = gf.grids[1] / 100
    data['Q'] *= 1000
    data['O3'] *= 1e+06
    data['N2O'] *= 1e+06
    data['CO'] *= 1e+06
    data['CH4'] *= 1e+06

    return pd.DataFrame({k: pd.Series(data[k], index=data['P'])
                         for k in data.keys()})
