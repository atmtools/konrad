#!/scratch/uni/u237/sw/anaconda36/envs/python36/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

#SBATCH --output=nlayers.log
#SBATCH --error=nlayers.log
#SBATCH --account=uni
#SBATCH --partition=uni-u237
#SBATCH --nodes=1-1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=lukas.kluft@gmail.com

"""Perform simulations to evaluate the impact of varying atmospheric layers.
"""
import multiprocessing
import time

import numpy as np
import typhon

import conrad


def use_levels(nlevels, season='tropical', experiment='nlayers', **kwargs):
    # Load the FASCOD atmosphere.
    gf = typhon.arts.xml.load(f'data/{season}.xml')
    gf = gf.extract_slice(slice(1, None), axis=1)  # omit bottom level.

    # Refine original pressure grid.
    p = conrad.utils.refined_pgrid(1013e2, 0.01e2, num=nlevels)
    gf.refine_grid(p, axis=1)

    # Create an atmosphere model.
    a = conrad.atmosphere.AtmosphereConvective.from_atm_fields_compact(gf)

    # # Create synthetic relative humidity profile.
    rh = conrad.utils.create_relative_humidity_profile(p, 0.75)
    a.relative_humidity = rh
    a.apply_H2O_limits()

    # Create a surface model.
    s = conrad.surface.SurfaceHeatCapacity.from_atmosphere(a, dz=5)

    # Create a radiation model.
    r = conrad.radiation.PSRAD(
        atmosphere=a,
        surface=s,
        daytime=1/np.pi,
        zenith_angle=20,
    )

    # Combine atmosphere and surface model into an RCE framework.
    rce = conrad.RCE(
        atmosphere=a, surface=s, radiation=r,
        delta=0.000,  # Run full number of itertations.
        timestep=0.2,  # 4.8 hour time step.
        writeevery=1.,  # Write netCDF output every day.
        max_iterations=5000,  # 1000 days maximum simulation time.
        outfile=f'results/{season}_{experiment}_{nlevels}_refined.nc'
    )

    # Start the actual simulation.
    rce.run()


if __name__ == '__main__':
    layers = [100, 150, 200, 250, 300, 350, 400, 600, 750, 1000]

    # The with block is not required for the model to run but prevents
    # creating and removing of symlinks during each iteration.
    with conrad.radiation.utils.PsradSymlinks():
        time.sleep(5)  # workaround for multiprocessing on  slow file systems.
        jobs = []
        for num in layers:
            p = multiprocessing.Process(
                target=use_levels,
                name='Layers-x{}'.format(num),
                args=[num],
            )
            jobs.append(p)
            p.start()

        # Wait for all jobs to finish before exit with-block.
        for job in jobs:
            job.join()
