"""Perform simulations with longwave bias correction. """
import logging
import multiprocessing
import os

import konrad
import netCDF4
import numpy as np
import xarray as xr


logger = logging.getLogger()


def bias_corrected_ecs(factor):
    ncfile = 'data/rce-rrtmg.nc'

    with netCDF4.Dataset(ncfile) as dataset:
        hmd = konrad.humidity.CoupledRH(
            p_tropo=dataset.variables['convective_top_plev'][-1],
        )

    atm = konrad.atmosphere.Atmosphere.from_netcdf(
        ncfile=ncfile,
        humidity=hmd,
        surface=konrad.surface.SurfaceHeatCapacity.from_netcdf(ncfile)
    )

    # Bias-corrected reference state.
    with xr.open_dataset('results/bias-correction/radiation-bias_1.nc') as ds:
        bias_correction = {k: ds[k].data for k in ds.data_vars}

    rce = konrad.RCE(
        atmosphere=atm,
        radiation=konrad.radiation.RRTMG(bias=bias_correction),
        delta=1e-5,  # Run full number of itertations.
        timestep=0.01,  # 4.8 hour time step.
        writeevery=10,
        max_iterations=25000,  # 1000 days maximum simulation time.
        outfile=f'results/rce-bias-corrected-new_1.nc',
    )
    rce.run()

    # Bias-corrected CO2 run.
    ncfile = f'results/bias-correction/radiation-bias_{factor}.nc'
    with xr.open_dataset(ncfile) as ds:
        bias_correction = {k: ds[k].data for k in ds.data_vars}

    atm['CO2'] *= factor

    rce = konrad.RCE(
        atmosphere=atm,
        radiation=konrad.radiation.RRTMG(bias=bias_correction),
        delta=1e-5,  # Run full number of itertations.
        timestep=0.2,  # 4.8 hour time step.
        writeevery=10.,
        max_iterations=25000,  # 1000 days maximum simulation time.
        outfile=f'results/rce-bias-corrected-new_{factor}.nc',
    )
    rce.run()


if __name__ == '__main__':
    # Factors to modify the CO2 concentration with.
    scale_factors = [2]

    nprocs = np.clip(
        a=len(scale_factors),
        a_min=1,
        a_max=int(os.environ.get('OMP_NUM_THREADS', 8))
    )

    with multiprocessing.Pool(nprocs) as p:
        p.map(bias_corrected_ecs, scale_factors)

    logger.info('All jobs finished.')
