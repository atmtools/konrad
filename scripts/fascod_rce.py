# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""Perform radiative-equilibirum simulations for the FASCOD atmospheres.
"""
import conrad
import typhon


fascod_seasons = [
    'subarctic-winter',
    'subarctic-summer',
    'midlatitude-winter',
    'midlatitude-summer',
    'tropical',
]

for season in fascod_seasons:
    # Load the FASCOD atmosphere.
    gf = typhon.arts.xml.load('data/{}.xml'.format(season))

    # Refine original pressure grid.
    p = typhon.math.nlogspace(1100e2, 0.1e2, 150)
    gf.refine_grid(p, axis=1)

    # Create an atmosphere model.
    a = conrad.atmosphere.AtmosphereFixedRH.from_atm_fields_compact(gf)

    # Create a sufrace model.
    s = conrad.surface.SurfaceAdjustableTemperature.from_atmosphere(a)

    # Create a sufrace model.
    r = conrad.radiation.PSRAD(atmosphere=a, surface=s)

    # Combine atmosphere and surface model into an RCE framework.
    rce = conrad.RCE(
        atmosphere=a,
        surface=s,
        radiation=r,
        delta=0.01,
        timestep=0.3,
        max_iterations=1000,
        outfile='results/{}.nc'.format(season)
        )

    # The with block is not required for the model to run but prevents
    # creating and removing of symlinks during each iteration.
    with conrad.radiation.utils.PsradSymlinks():
        rce.run()  # Start simulation.
