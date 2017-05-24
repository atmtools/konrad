# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""Perform simulations to evaluate the impact of varying CO2 concentrations.
"""
import multiprocessing
import time

import typhon

import conrad


def scale_co2(factor, **kwargs):
    # Load the FASCOD atmosphere.
    gf = typhon.arts.xml.load('data/tropical.xml')
    gf = gf.extract_slice(slice(1, None), axis=1)  # omit bottom level.

    # Refine original pressure grid.
    p = typhon.math.nlogspace(1013e2, 0.1e2, 200)
    gf.refine_grid(p, axis=1)

    # Create an atmosphere model.
    a = conrad.atmosphere.AtmosphereConvective.from_atm_fields_compact(gf)
    a['CO2'] *= factor

    # Create synthetic relative humidity profile.
    # a['T'] -= 30
    a.relative_humidity = conrad.utils.create_relative_humidity_profile(p, 0.75)

    # Create a sufrace model.
    # s = conrad.surface.SurfaceAdjustableTemperature.from_atmosphere(a)
    s = conrad.surface.SurfaceCoupled.from_atmosphere(a)

    # Create a sufrace model.
    r = conrad.radiation.PSRAD(atmosphere=a, surface=s)

    # Combine atmosphere and surface model into an RCE framework.
    rce = conrad.RCE(
        atmosphere=a,
        surface=s,
        radiation=r,
        delta=0.000,
        timestep=0.0625,
        writeevery=1.,
        max_iterations=5000,
        outfile='results/co2_x{}.nc'.format(factor)
    )

    # The with block is not required for the model to run but prevents
    # creating and removing of symlinks during each iteration.
    with conrad.radiation.utils.PsradSymlinks():
        rce.run()  # Start simulation.


if __name__ == '__main__':
    scale_factors = [0.5, 1, 2, 4, 8]

    with conrad.radiation.utils.PsradSymlinks():
        time.sleep(5)
        jobs = []
        for factor in scale_factors:
            p = multiprocessing.Process(
                target=scale_co2,
                name='CO2-x{}'.format(factor),
                args=[factor],
            )
            jobs.append(p)
            p.start()

        # Wait for all jobs to finish before exit with-block.
        for job in jobs:
            job.join()
