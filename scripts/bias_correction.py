"""Perform simulations with longwave bias correction. """
import logging
import multiprocessing
import os

import conrad
import netCDF4
import numpy as np
from typhon.arts import xml


logger = logging.getLogger()


def scale_co2(factor):
    ncfile = conrad.utils.get_filepath(
        'rce-rrtmg', 'scale-co2-coupledrh', factor)

    sfc = conrad.surface.SurfaceHeatCapacity.from_netcdf(ncfile)

    with netCDF4.Dataset(ncfile) as dataset:
        hmd = conrad.humidity.CoupledRH(
            p_tropo=dataset.variables['convective_top_plev'][-1],
        )

    atm = conrad.atmosphere.Atmosphere.from_netcdf(
        ncfile=ncfile,
        humidity=hmd,
        surface=sfc,
    )

    bias_correction = xml.load(f'results/co2_bias/bias_{float(factor)}.xml')

    # Combine all submodels into a RCE framework.
    rce = conrad.RCE(
        atmosphere=atm,
        radiation=conrad.radiation.RRTMG(bias={'net_htngrt': bias_correction}),
        delta=1e-5,  # Run full number of itertations.
        timestep=0.2,  # 4.8 hour time step.
        writeevery=10.,  # Write netCDF output every day.
        max_iterations=20000,  # 1000 days maximum simulation time.
        outfile=f'results/rce-bias_correction_{factor}.nc',
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
