# -*- coding: utf-8 -*-

"""Perform radiative-equilibirum test simulations.

"""

import os
import numpy as np
import copy
import time
import multiprocessing
from typhon.physics import e_eq_water_mk
from scipy.interpolate import interp1d

import conrad
from conrad import constants


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

def ozonesquash(a, squash, normalise=False):
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

    if normalise:
        print(np.diff(a['z'][:].data.shape))
        dz = np.hstack((a['z'][0, 0].data-a.surface.height, np.diff(z))) #this is half a level out...
        print(dz.shape)
        totalo3_old = np.sum(o3*dz)
        totalo3_new = np.sum(new_o3*dz)
        new_o3 *= totalo3_old / totalo3_new

    return new_o3

def moistlapse(atmosphere):
    T = atmosphere['T'][0, :]
    p = atmosphere['plev'][:]
    phlev = atmosphere['phlev'][:]
    # Use short formula symbols for physical constants.
    g = constants.earth_standard_gravity
    L = constants.heat_of_vaporization
    #L = 2500800 - 2360*(T-273) + 1.6*(T-273)**2 - 0.06*(T-273)**3
    Rd = constants.specific_gas_constant_dry_air
    Rv = constants.specific_gas_constant_water_vapor
    epsilon = constants.gas_constant_ratio
    Cp = constants.isobaric_mass_heat_capacity

    gamma_d = g / Cp  # dry lapse rate
    q_saturated = epsilon * e_eq_water_mk(T) / p

    gamma_m = (gamma_d * ((1 + (L * q_saturated) / (Rd * T)) /
                          (1 + (L**2 * q_saturated) / (Cp * Rv * T**2))
                          )
    )
    lapse = interp1d(p, gamma_m, fill_value='extrapolate')(phlev[:-1])
    return lapse


def scaleozone(co2factor, squash, w, hardadj=False): #sst=290,
    squashname = str(squash).replace('.', '')
    co2name = str(co2factor).replace('.', '')
    wname = str(w).replace('.', '')

    if hardadj:
        myfile = '/scratch/local1/m300580/conrad/rrtmg_sun/hardadj/w-0/CO2-1/dirunal_lat0_and_surfacesink717/200levels_squash{}.nc'.format(squashname)
    else:
        myfile = '/scratch/local1/m300580/conrad/rrtmg_sun/rlxadj/w-0/CO2-1/dirunal_lat0_and_surfacesink726/200levels_squash{}.nc'.format(squashname)

    a = conrad.atmosphere.Atmosphere.from_netcdf(myfile,
                                                 humidity=conrad.humidity.FixedRH(rh_tropo=0), #, vmr_strato=4*10**-6, transition_depth=1),
                                                 convection=conrad.convection.RelaxedAdjustment(),
                                                 lapse=conrad.lapserate.MoistLapseRate(),
                                                 #upwelling=conrad.upwelling.StratosphericUpwelling(w=w) #w=17.28)
                                                 surface=conrad.surface.SurfaceHeatSink.from_netcdf(myfile, heat_flux=72.6, rho=1025, cp=3850, dz=50)
                                                 )
    if w == 0:
        a.attrs.update({'upwelling': conrad.upwelling.NoUpwelling()})

    # Refine the pressure grid to cover 200 vertical levels.
    pgrid = conrad.utils.refined_pgrid(1013e2, 0.01e2, 200)
    a = a.refine_plev(pgrid)

#    ds = Dataset(myfile)
#    ll = np.min(np.where(ds['net_htngrt'][-1, :] > -0.0001))
#    a.attrs.update({'upwelling': conrad.upwelling.StratosphericUpwelling(w=w, lowest_level=ll)})

    if hardadj:
        a.attrs.update({'convection': conrad.convection.HardAdjustment()})
    else:
        tau0 = 1/24 # 1 hour
        tau = tau0*np.exp(101300 / pgrid)
        a.convection.convective_tau = tau

#    a.lapse.lapserate = moistlapse(a)

    a.tracegases_rcemip()
    a['O3'].values = ozonesquash(a, squash)[np.newaxis, :]

    # Other atmospheric changes
    a['CO2'] *= co2factor


    # RRTMG radiation
    #rad = conrad.radiation.RRTMG(zenith_angle=47.88)
    rad = conrad.radiation.RRTMG(zenith_angle=0, solar_constant=1360.85, diurnal_cycle=True)

    # folder to save files in
    if hardadj:
        filepath = '/scratch/local1/m300580/conrad/rrtmg_sun/hardadj/w-{}/CO2-{}/dirunal_lat0_and_surfacesink717/'.format(wname, co2name)
    else:
        filepath = '/scratch/local1/m300580/conrad/rrtmg_sun/rlxadj/w-{}/CO2-{}/dirunal_lat0_and_surfacesink726/'.format(wname, co2name)#50levels_squash{}.nc'.format(squashname)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Combine atmosphere and surface model into an RCE framework.
    rce = conrad.RCE(
        atmosphere=a,
        radiation=rad,
        delta=0,#0.001,
        timestep=0.01, #6/24, #0.5, #3/24, #1, #, #0.2, #0.02,
        writeevery=1, #1000,
        max_iterations=1000, #600000,
        outfile=filepath+'200levels_squash{}_10days.nc'.format(squashname)
        )

    # The with block is not required for the model to run but prevents
    # creating and removing of symlinks during each iteration.
    with conrad.radiation.utils.PsradSymlinks():
        rce.run()  # Start simulation.

#if __name__ == '__main__':
#
#    squashfactors = [0.7, 0.85, 1.0, 1.15, 1.3]
#
#    co2factors = [0.25, 0.5, 1, 2, 4]
#    # The with block is not required for the model to run but prevents
#    # creating and removing of symlinks during each iteration.
#    with conrad.radiation.utils.PsradSymlinks():
#        time.sleep(5)  # workaround for multiprocessing on slow file systems.
#        for adj in [True, False]:
#            for w in 0, 0.2, 0.5:
#                for o3 in squashfactors:
#                    jobs = []
#                    for co2 in co2factors:
#                        p = multiprocessing.Process(target=scaleozone,
#                                                    args=[co2, o3, w, adj],
#                                                    )
#                        jobs.append(p)
#                        p.start()
#
#                    # Wait for all jobs to finish before exit with-block.
#                    for job in jobs:
#                        job.join()


if __name__ == '__main__':

    squashfactors = [0.7, 0.85, 1.0, 1.15, 1.3]

    co2factors = [1]#, 0.25, 0.5, 2, 4]
    time.sleep(5)  # workaround for multiprocessing on slow file systems.
    for co2 in co2factors:
        jobs = []
        for o3 in squashfactors:
            p = multiprocessing.Process(target=scaleozone,
                                        args=[co2, o3, 0, False],
                                        )
            jobs.append(p)
            p.start()

        # Wait for all jobs to finish before exit with-block.
        for job in jobs:
            job.join()
