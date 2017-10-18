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
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=lukas.kluft@gmail.com

"""Perform simulations to evaluate the impact of varying atmospheric layers.
"""
import logging
import multiprocessing

import typhon

import conrad


logger = logging.getLogger()


def use_levels(nlevels, atmosphere='tropical-standard',
               experiment='nlayers-refined-fix', **kwargs):
    # Load the FASCOD atmosphere.
    gf = typhon.arts.xml.load(f'data/{atmosphere}.xml')

    # Refine original pressure grid.
    p = conrad.utils.refined_pgrid(1013e2, 0.01e2, num=nlevels, shift=0.5)
    gf.refine_grid(p, axis=1, fill_value='extrapolate')

    # # Load atmosphere and surface model from netCDF file.
    # # The models used as initial state are in equilibrium to avoid signals
    # # from pure adjustment in the results.
    # ncfile = conrad.utils.get_filepath(
    #     atmosphere=atmosphere,
    #     experiment='nlayers-convective',
    #     scale=nlevels,
    # )
    # a = conrad.atmosphere.AtmosphereConvective.from_netcdf(ncfile)
    # s = conrad.surface.SurfaceHeatCapacity.from_netcdf(ncfile, dz=100)
    # a['CO2'] *= 2  # Scale the CO2 concentration.

    a = conrad.atmosphere.AtmosphereConvective.from_atm_fields_compact(gf)
    s = conrad.surface.SurfaceHeatCapacity.from_atmosphere(a)

    # Create synthetic relative humidity profile.
    rh = conrad.utils.create_relative_humidity_profile(p, 0.75)
    a.relative_humidity = rh
    a.apply_H2O_limits()

    # Create a radiation model.
    r = conrad.radiation.PSRAD(atmosphere=a, surface=s)

    # Combine all submodels into a RCE framework.
    rce = conrad.RCE(
        atmosphere=a, surface=s, radiation=r,
        delta=0.000,  # Run full number of itertations.
        timestep=0.2,  # 4.8 hour time step.
        writeevery=1.,  # Write netCDF output every day.
        max_iterations=15000,  # 1000 days maximum simulation time.
        outfile=conrad.utils.get_filepath(atmosphere, experiment, nlevels),
    )

    # Start the actual simulation.
    rce.run()


if __name__ == '__main__':
    # Number of pressure gridpoints to use for different runs.
    level_numbers = range(100, 1001, 100)

    # The with block is not required for the model to run but prevents
    # creating and removing of symlinks during each iteration.
    with conrad.radiation.utils.PsradSymlinks():
        pool = multiprocessing.Pool(8)
        pool.map(use_levels, level_numbers)

    logger.info('All jobs finished.')
