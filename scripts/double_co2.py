#!/scratch/uni/u237/sw/anaconda36/envs/python36/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

#SBATCH --output=double_co2.log
#SBATCH --error=double_co2.log
#SBATCH --account=uni
#SBATCH --partition=uni-u237
#SBATCH --nodes=1-1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=lukas.kluft@gmail.com

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
    a = conrad.atmosphere.AtmosphereFixedVMR.from_atm_fields_compact(gf)
    a['CO2'] *= factor

    # # Create synthetic relative humidity profile.
    # rh = conrad.utils.create_relative_humidity_profile(p, 0.75)
    # a.relative_humidity = rh
    # a.adjust_vmr()

    # Create a sufrace model.
    s = conrad.surface.SurfaceAdjustableTemperature.from_atmosphere(a)

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
        max_iterations=3200,
        outfile='results/tropical_co2_x{}-re.nc'.format(factor)
    )

    rce.run()  # Start simulation.


if __name__ == '__main__':
    scale_factors = [0.5, 1, 2, 4, 8]

    # The with block is not required for the model to run but prevents
    # creating and removing of symlinks during each iteration.
    with conrad.radiation.utils.PsradSymlinks():
        time.sleep(5)  # workaround for multiprocessing on  slow file systems.
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
