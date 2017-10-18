#!/scratch/uni/u237/sw/anaconda36/envs/python36/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

#SBATCH --output=simple.log
#SBATCH --error=simple.log
#SBATCH --account=uni
#SBATCH --partition=uni-u237
#SBATCH --nodes=1-1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=lukas.kluft@gmail.com

"""Perform radiative-equilibirum test simulations.

Note:
    This script is meant for debugging and testing. It should **not** be used
    as an example, as it might include functionality which is under
    development.
"""
import typhon

import conrad


# Load the FASCOD atmosphere.
gf = typhon.arts.xml.load('data/tropical-standard.xml')

# Refine original pressure grid.
p = conrad.utils.refined_pgrid(1013e2, 0.1e2, 800)
gf.refine_grid(p, axis=1)

# Create an atmosphere model.
a = conrad.atmosphere.AtmosphereMoistConvective.from_atm_fields_compact(gf)

# Create synthetic relative humidity profile.
a.relative_humidity = conrad.utils.create_relative_humidity_profile(p, 0.75)

# Create a sufrace model.
# s = conrad.surface.SurfaceAdjustableTemperature.from_atmosphere(a)
s = conrad.surface.SurfaceHeatCapacity.from_atmosphere(a)

# Create a sufrace model.
r = conrad.radiation.PSRAD(atmosphere=a, surface=s)

# Combine atmosphere and surface model into an RCE framework.
rce = conrad.RCE(
    atmosphere=a,
    surface=s,
    radiation=r,
    delta=0.000,
    timestep=0.25,
    writeevery=10.,
    max_iterations=5000,
    outfile='results/renew_simple_800.nc'
    )

# The with block is not required for the model to run but prevents
# creating and removing of symlinks during each iteration.
with conrad.radiation.utils.PsradSymlinks():
    rce.run()  # Start simulation.
