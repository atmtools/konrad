# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""Perform radiative-equilibirum test simulations.

Note:
    This script is meant for debugging and testing. It should **not** be used
    as an example, as it might include functionality which is under
    development.
"""
import conrad
import typhon
import numpy as np
import copy
import time
import multiprocessing

def ozoneshift(a, shift):
    """
    Shifts the ozone profile upwards or downwards, without changing the shape
    
    Parameters:
        a: atmosphere
        shift: number of atmospheric levels to shift the profile by
    """
    o3 = a['O3'][0, :].data
    new_o3 = np.roll(o3, shift)
    for level in range(0, shift):
        new_o3[level] = new_o3[shift]
    return new_o3

def ozonesquash(a, squash):
    """
    Squashes the ozone profile upwards or stretches it downwards, with no 
        change to the shape of the profile above the ozone concentration maximum
    Parameters:
        a: atmosphere
        squash: float, with 1 being no squash, 
            numbers < 1 squashing the profile towards the maximum,
            numbers > 1, stretching the profile downwards
    """
    o3 = a['O3'][0, :].data
    z = a['z'][0, :].data
    i_max_o3 = np.argmax(o3)
    
    sqz = (z - z[i_max_o3])*squash + z[i_max_o3]
    new_o3 = copy.copy(o3)
    new_o3[:i_max_o3] = np.interp(z[:i_max_o3], sqz, o3)
    return new_o3

def moistlapse(a):
    """ 
    Calculates the moist lapse rate from the atmospheric temperature and 
        humidity profiles 
    Parameters:
        a: atmosphere
    """
    g = conrad.constants.g
    Lv = 2501000
    R = 287
    eps = 0.62197
    Cp = conrad.constants.isobaric_mass_heat_capacity
    VMR = a['H2O'][0, :]
    T = a['T'][0, :]
    lapse = g*(1 + Lv*VMR/R/T)/(Cp + Lv**2*VMR*eps/R/T**2)
    return lapse.values

#for levels in [200]: #[100, 200, 300, 400, 500]:

#for squash in np.arange(0.85, 1.3, 0.05): # range(0, 32, 2):
def scaleozone(squash, levels):
    squashname = str(squash).replace('.', '')
    
    # Load the FASCOD atmosphere.
    #gf = typhon.arts.xml.load('/home/mpim/m300580/conrad/scripts/data/tropical.xml')
    #gf = gf.extract_slice(slice(1, None), axis=1)  # omit bottom level
    
    gf = typhon.arts.xml.load('/home/mpim/m300580/conrad/scripts/data/tropical-convective.xml')
    #gf = typhon.arts.xml.load('/home/mpim/m300580/conrad/scripts/data/tropical-fixedRH.xml')
    # Refine original pressure grid.
    p = conrad.utils.refined_pgrid(1013e2, 0.01e2, levels)
    gf.refine_grid(p, axis=1)
    
#    # Refine original pressure grid.
#    p_old = gf.grids[1]
#    p = typhon.math.nlogspace(p_old.max(), p_old.min(), levels)
#    gf.refine_grid(p, axis=1)
    
    # Create an atmosphere model.
    # Lapse rate varies with height, but fixed throughout run
    a = conrad.atmosphere.AtmosphereConvective.from_atm_fields_compact(gf, lapse=0.0065*np.ones((1, levels)))
    #a = conrad.atmosphere.AtmosphereFixedRH.from_atm_fields_compact(gf)
    #a = conrad.atmosphere.AtmosphereSlowConvective.from_atm_fields_compact(gf, tau=np.ones((1, levels))) #, lapse=0.0065*np.ones((1, levels)))
    #a['lapse'][0] = moistlapse(a)
    #a['convective_tau'] *= 2
    #a['convective_tau'][0, :40] = 0
    
    # If the lapse rate should change at each time step depending on the
    # updated atmospheric temperature and humidity use AtmosphereMoistConvective
    
    # Create a sufrace model.
    #s = conrad.surface.SurfaceHeatCapacity.from_atmosphere(a, dz=3)
    #s = conrad.surface.SurfaceFixedTemperature.from_netcdf('/home/mpim/m300580/Documents/lukas/eq_atmospheres_netcdf/tropical-convective.nc')#from_atmosphere(a)
    s = conrad.surface.SurfaceHeatCapacity.from_netcdf('/home/mpim/m300580/Documents/lukas/eq_atmospheres_netcdf/tropical-convective.nc')#from_atmosphere(a)
    #s['temperature'] = 320
    
    #TODO a['convective_timescale'] = some array, a function of height 
                                        #default zero
    
    # Other atmospheric changes
    #a['CO2'] *= 0.5
    a['O3'].values = ozonesquash(a, squash)[np.newaxis, :]
    #a['O3'] *= factor
    
    # Create synthetic relative humidity profile.
    rh = conrad.utils.create_relative_humidity_profile(p, 0.75)
    a.relative_humidity = rh
    
    # Create a sufrace model.
    r = conrad.radiation.PSRAD(atmosphere=a, surface=s, zenith_angle=20) #daytime=0.3, 
    
    # Combine atmosphere and surface model into an RCE framework.
    rce = conrad.RCE(
        atmosphere=a,
        surface=s,
        radiation=r,
        delta=0,#0.001,
        timestep=0.02, #0.02,
        writeevery=1,
        max_iterations=20000,
        #outfile='/home/mpim/m300580/conrad/scripts/results/readjust_test.nc'
        outfile='/scratch/local1/m300580/conrad/ozonesquash/con_{}levels/squash{}.nc'.format(levels, squashname)
        )
    
    # The with block is not required for the model to run but prevents
    # creating and removing of symlinks during each iteration.
    with conrad.radiation.utils.PsradSymlinks():
        rce.run()  # Start simulation.

if __name__ == '__main__':
    squashfactors = np.arange(0.7, 1.35, 0.05)
    #squashfactors = [0.7, 0.75]
    levelset = [400, 600, 1000]
    # The with block is not required for the model to run but prevents
    # creating and removing of symlinks during each iteration.
    with conrad.radiation.utils.PsradSymlinks():
        time.sleep(5)  # workaround for multiprocessing on slow file systems.
        jobs = []
        for factor in squashfactors:
            p = multiprocessing.Process(target=scaleozone, args=[factor, 600],)
            jobs.append(p)
            p.start()

        # Wait for all jobs to finish before exit with-block.
        for job in jobs:
            job.join()
