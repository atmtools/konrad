# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""Perform radiative-equilibirum simulations for different temperature profiles
that are shifted against each other.
"""
import konrad
import typhon


# Load the FASCOD atmosphere.
gf = typhon.arts.xml.load('data/tropical.xml')

# Refine original pressure grid.
p = typhon.math.nlogspace(1100e2, 0.1e2, 150)
gf.refine_grid(p, axis=1)

for shift in range(-30, 31, 15):
    # Create an atmosphere model.
    a = konrad.atmosphere.AtmosphereFixedRH.from_atm_fields_compact(gf)

    # Make isothermal atmosphere.
    rh = typhon.atmosphere.relative_humidity(a['H2O'], a['plev'], a['T'])
    a['T'] += shift
    a.relative_humidity = rh
    a.apply_H2O_limits()

    # Create a sufrace model.
    s = konrad.surface.SurfaceAdjustableTemperature.from_atmosphere(a)

    # Create a sufrace model.
    r = konrad.radiation.PSRAD(atmosphere=a, surface=s)

    # Combine atmosphere and surface model into an RCE framework.
    rce = konrad.RCE(
        atmosphere=a,
        surface=s,
        radiation=r,
        delta=0.01,
        timestep=0.1,
        max_iterations=10000,
        outfile=f'results/shifted_{shift:+d}.nc'
        )

    # The with block is not required for the model to run but prevents
    # creating and removing of symlinks during each iteration.
    with konrad.radiation.utils.PsradSymlinks():
        rce.run()  # Start simulation.
