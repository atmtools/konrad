# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""Perform radiative-equilibirum simulations for the FASCOD atmospheres.
"""
import conrad as c
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
    atmosphere = c.atmosphere.AtmosphereFixedRH.from_atm_fields_compact(gf)
    atmosphere['O3'] *= 0.01  # TODO: Dirty workaround!

    # # Create a sufrace model.
    # surface = c.surface.SurfaceAdjustableTemperature.from_atmosphere(
    #     atmosphere)

    # # Create a sufrace model.
    # radiation = c.radiation.PSRAD(atmosphere, surface)

    # Combine atmosphere and surface model into an RCE framework.
    model = c.ConRad(atmosphere=atmosphere,
                     # surface=surface,
                     # radiation=radiation,
                     dt=1,
                     max_iterations=500,
                     outfile='results/{}.nc'.format(season)
                     )

    with c.radiation.utils.PsradSymlinks():
        model.run()  # Start simulation.
