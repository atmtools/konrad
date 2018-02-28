#!/usr/bin/env python3
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
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=lukas.kluft@gmail.com

"""Perform simulations to evaluate the impact of varying CO2 concentrations.
"""
import logging
import multiprocessing
import os

import conrad
import netCDF4
import numpy as np


logger = logging.getLogger()


def scale_co2(factor, atmosphere='rce-rrtmg',
              experiment='scale-co2-coupledrh'):
    # Load atmosphere and surface in equilibrium to prevent signals from
    # adjustment to model physics.
    ncfile = f'data/{atmosphere}.nc'
    surface = conrad.surface.SurfaceHeatCapacity.from_netcdf(ncfile)

    with netCDF4.Dataset(ncfile) as dataset:
        humidity = conrad.humidity.CoupledRH(
            p_tropo=dataset.variables['convective_top_plev'][-1],
        )

    atmosphere = conrad.atmosphere.Atmosphere.from_netcdf(
        ncfile=ncfile,
        humidity=humidity,
        surface=surface,
    )

    # Scale the CO2 concentration.
    atmosphere['CO2'] *= factor

    # Combine all submodels into a RCE framework.
    rce = conrad.RCE(
        atmosphere=atmosphere,
        radiation=conrad.radiation.RRTMG(),
        delta=1e-5,  # Run full number of itertations.
        timestep=0.05,  # 4.8 hour time step.
        writeevery=10.,  # Write netCDF output every day.
        max_iterations=80000,  # 1000 days maximum simulation time.
        outfile=conrad.utils.get_filepath(atmosphere, experiment, factor),
    )

    # Start actual simulation.
    rce.run()


if __name__ == '__main__':
    # Factors to modify the CO2 concentration with.
    scale_factors = [0.5, 1, 2, 4, 8]

    nprocs = np.clip(
        a=len(scale_factors),
        a_min=1,
        a_max=os.environ.get('OMP_NUM_THREADS', 8)
    )

    with multiprocessing.Pool(nprocs) as p:
        p.map(scale_co2, scale_factors)

    logger.info('All jobs finished.')
