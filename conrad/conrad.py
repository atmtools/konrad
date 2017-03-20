# -*- coding: utf-8 -*-
"""

"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from typhon import atmosphere

from . import utils
from . import plots
from . import core


logger = logging.getLogger()

__all__ = [
    'ConRad',
]

atmvariables = {
    'CH4': ('Methane', 'VMR'),
    'CO': ('Carbon monoxide', 'VMR'),
    'CO2': ('Carbon dioxide', 'VMR'),
    'N2O': ('Nitrogen', 'VMR'),
    'O3': ('Ozone', 'VMR'),
    'Q': ('Water vapor', 'VMR'),
    'T': ('Temperature', 'K'),
}


class ConRad():
    """Implementation of a radiative-convective equilibrium model.

    Examples:
        Create an object to setup a simulation.
        >>> c = ConRad(data=pands.DataFrame(...))
        >>> c.run()
    """
    def __init__(self, sounding=None, fix_rel_humidity=True, outfile=None,
                 convective_adjustment=False, dt=1, max_iterations=5000,
                 delta=0.03):
        """Set-up a radiative-convective model.

        Parameters:
            sounding (pd.DataFrame): pandas DataFrame representing an
                atmospheric sounding.
            fix_rel_humidity (bool): Adjust the water vapor mixing ratio to
                keep the relative humidity constant.
            outfile (str): netCDF4 file to store output.
            convective_adjustment (bool): Adjust the temperature profile to an
                critical lapse rate.
            dt (float): Time step in days.
            max_iterations (int): Maximum number of iterations.
            delta (float): Stop criterion. If the heating rate is below this
                threshold for all levels, skip further iterations.
        """
        self.convective_adjustment = convective_adjustment
        self.delta = delta
        self.dt = dt  # TODO: Decide: days or hours
        self.outfile = outfile
        self.fix_rel_humidity = fix_rel_humidity
        self.max_iterations = max_iterations
        self.niter = 0
        self.rad_lw = None
        self.rad_sw = None
        self.sounding = sounding

        logging.info('Created ConRad object: {}'.format(self))

    def plot_sounding_p(self, variable, ax=None, **kwargs):
        return plots.atmospheric_profile_p(
                self.sounding.index, self.sounding[variable], **kwargs)

    def plot_sounding_z(self, variable, ax=None, **kwargs):
        return plots.atmospheric_profile_z(
                self.sounding['Z'], self.sounding[variable], **kwargs)

    def plot_overview_z(self, axes=None, **kwargs):
        return plots.plot_overview_z(
            self.sounding, self.rad_lw, self.rad_sw, axes, **kwargs)

    def plot_overview_p(self, axes=None, **kwargs):
        return plots.plot_overview_z(
            self.sounding, self.rad_lw, self.rad_sw, axes, **kwargs)

    @utils.with_psrad_symlinks
    def run(self):
        """Run the radiative-convective equilibirum model."""
        from . import psrad

        logger.info(
            'Start iterative model run.\n'
            'Maximum number of iterations: {}\n'
            'Stop criterion: {}'.format(self.max_iterations, self.delta)
            )

        # Create netCDF4 file to store simulation results.
        if self.outfile is not None:
            utils.create_netcdf(
                self.outfile,
                pressure=self.sounding['P'],
                description='Radiative-convective equilibrium simulation.',
                variable_description=atmvariables,
                )

        while self.niter < self.max_iterations:
            logger.debug('Enter iteration {}.'.format(self.niter))

            # TODO: Maybe merge all the radiative quantities?
            # Calculate shortwave and longwave heating rates.
            self.rad_lw = psrad.psrad_lw(self.sounding)
            self.rad_sw = psrad.psrad_sw(self.sounding)

            T_new = core.adjust_temperature(
                self, convective_adjustment=self.convective_adjustment)

            # TODO: I do not like this solution. Maybe switch to a boolean
            # `converged` or so?
            dT = T_new - self.sounding['T']

            if self.fix_rel_humidity:
                logger.debug('Adjust VMR to preserve relative humidity.')
                self.sounding['Q'] = core.adjust_vmr(self.sounding, T_new)

            self.sounding['T'] = T_new

            if self.outfile is not None:
                utils.append_timestep_netcdf(
                    self.outfile,
                    # TODO: Dirty hack for now. Not all variables are supported
                    # for appending to netCDF4 . Therefore only the ones with
                    # proper definition are passed to the function.
                    {v: self.sounding[v].values for v in atmvariables},
                    self.niter * 24 * self.dt,
                    )

            if np.all(np.abs(dT) < self.delta):
                logger.info(
                    'Converged after {} iterations.'.format(self.niter))
                break

            self.niter += 1
        else:
            logger.info('Stopped after maximum number of iterations.')
