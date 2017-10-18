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
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=lukas.kluft@gmail.com

"""Perform simulations to evaluate the impact of varying CO2 concentrations.
"""
import logging
import multiprocessing

from typhon.math import nlogspace

import conrad


logger = logging.getLogger()


def scale_co2(factor, atmosphere='secondmax',
              experiment='fixed-rh', **kwargs):
    # Load atmosphere and surface model from netCDF file.
    # The models used as initial state are in equilibrium to avoid signals
    # from pure adjustment in the results.
    ncfile = f'data/{atmosphere}.nc'
    a = conrad.atmosphere.AtmosphereMoistConvective.from_netcdf(ncfile)
    a = a.refine_plev(conrad.utils.refined_pgrid(1013e2, 0.01e2, 250))
    s = conrad.surface.SurfaceHeatCapacity.from_netcdf(ncfile, dz=50)

    # Scale the CO2 concentration.
    a['CO2'] *= factor

    # # Create synthetic relative humidity profile.
    # rh = conrad.utils.create_relative_humidity_profile(a['plev'].values)
    # a.relative_humidity = rh
    # a.apply_H2O_limits()

    # Create a radiation model.
    r = conrad.radiation.PSRAD(atmosphere=a, surface=s)

    # Combine all submodels into a RCE framework.
    rce = conrad.RCE(
        atmosphere=a, surface=s, radiation=r,
        delta=0.000,  # Run full number of itertations.
        timestep=0.15,  # 4.8 hour time step.
        writeevery=1.,  # Write netCDF output every day.
        max_iterations=10000,  # 1000 days maximum simulation time.
        outfile=conrad.utils.get_filepath(atmosphere, experiment, factor),
    )

    # Start actual simulation.
    rce.run()


if __name__ == '__main__':
    # Factors to modify the CO2 concentration with.
    scale_factors = [0.5, 1, 2, 4]

    # The with block is not required for the model to run but prevents
    # creating and removing of symlinks during each iteration.
    with conrad.radiation.utils.PsradSymlinks():
        p = multiprocessing.Pool(4)
        p.map(scale_co2, scale_factors)

    logger.info('All jobs finished.')
