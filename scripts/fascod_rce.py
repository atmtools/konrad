# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""Perform radiative-equilibirum simulations for the FASCOD atmospheres.
"""
import conrad
from conrad import (atmosphere, surface)
import typhon


fascod_seasons = [
    # 'subarctic-winter',
    # 'subarctic-summer',
    # 'midlatitude-winter',
    # 'midlatitude-summer',
    'tropical',
]

for season in fascod_seasons:
    # Load the FASCOD atmosphere.
    gf = typhon.arts.xml.load('data/{}.xml'.format(season))

    # Refine original pressure grid.
    p = typhon.math.nlogspace(1100e2, 0.1e2, 75)
    gf.refine_grid(p, axis=1)

    # Create an atmosphere model.
    atmosphere = atmosphere.AtmosphereFixedRH.from_atm_fields_compact(gf)
    atmosphere['O3'] *= 0.01  # TODO: Dirty workaround!

    # Create a sufrace model.
    surface = surface.SurfaceAdjustableTemperature.from_atmosphere(atmosphere)

    # Combine atmosphere and surface model into an RCE framework.
    c = conrad.ConRad(atmosphere=atmosphere,
                      surface=surface,
                      dt=1,
                      max_iterations=500,
                      outfile='results/{}.nc'.format(season)
                      )

    c.run()  # Start simulation.
