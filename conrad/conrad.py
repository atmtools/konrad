# -*- coding: utf-8 -*-
"""Implementation of a radiative-convective equilibrium model (RCE).
"""
import logging

import numpy as np

from . import utils


logger = logging.getLogger()

__all__ = [
    'RCE',
]


class RCE():
    """Implementation of a radiative-convective equilibrium model.

    Examples:
        Create an object to setup and run a simulation:
        >>> c = RCE(atmosphere=a, surface=s, radiation=r)
        >>> c.run()
    """
    def __init__(self, atmosphere, surface, radiation, outfile=None,
                 timestep=1, delta=0.01, writeevery=1, max_iterations=5000):
        """Set-up a radiative-convective model.

        Parameters:
            atmosphere (Atmosphere): `conrad.atmosphere.Atmosphere`.
            surface (Surface): An surface object inherited from
                `conrad.surface.Surface`.
            outfile (str): netCDF4 file to store output.
            writevery(int or float): Set frequency in which to write output.
                int: Every nth timestep is written.
                float: Every nth day is written.
            timestep (float): Iteration time step in days.
            max_iterations (int): Maximum number of iterations.
            delta (float): Stop criterion. If the heating rate is below this
                threshold for all levels, skip further iterations.
        """
        # Sub-models.
        self.atmosphere = atmosphere
        self.surface = surface
        self.radiation = radiation

        # Control parameters.
        self.delta = delta
        self.timestep = timestep
        self.writeevery = writeevery
        self.max_iterations = max_iterations

        # TODO: Maybe delete? One could use the return value of the radiation
        # model directly.
        self.heatingrates = None

        # Internal variables.
        self.converged = False
        self.niter = 0

        self.outfile = outfile

        logging.info('Created ConRad object:\n{}'.format(self))

    def __repr__(self):
        retstr = '{}(\n'.format(self.__class__.__name__)
        # Loop over all public object attributes.
        for a in filter(lambda k: not k.startswith('_'), self.__dict__):
            retstr += '    {}={},\n'.format(a, getattr(self, a))
        retstr += ')'

        return retstr

    def get_hours_passed(self):
        """Return the number of house passed since model start.

        Returns:
            float: Hours passed since model start.
        """
        return self.niter * 24 * self.timestep

    def calculate_heatingrates(self):
        """Use the radiation sub-model to calculate heatingrates."""
        self.heatingrates = self.radiation.get_heatingrates(
            atmosphere=self.atmosphere,
            surface=self.surface,
            )

    def is_converged(self):
        """Check if the atmosphere is in radiative-convective equilibrium.

        Returns:
            bool: ``True`` if converged, else ``False``.
        """
        return np.all(np.abs(self.atmosphere['deltaT']) < self.delta)

    def check_if_write(self):
        """Check if current timestep should be appended to output netCDF.

        Do not write, if no output file is specified.

        Returns:
            bool: True, if timestep should be written.
        """
        if self.outfile is None:
            return False

        if isinstance(self.writeevery, int):
            return self.niter % self.writeevery == 0
        elif isinstance(self.writeevery, float):
            # Add `0.5 * dt` to current timestep to make float comparison more
            # robust. Otherwise `3.3 % 3 < 0.3` is True.
            r = (((self.niter + 0.5) * self.timestep) % self.writeevery)
            return r < self.timestep
        else:
            raise TypeError('Only except input of type `float` or `int`.')

    # TODO: Consider implementing a more powerful netCDF usage. Currently
    # storing values from different sub-modules will be difficult.
    def create_outfile(self):
        """Create netCDF4 file to store simulation results."""
        data = self.atmosphere.merge(self.heatingrates, overwrite_vars='H2O')
        data.to_netcdf(self.outfile,
                       mode='w',
                       unlimited_dims=['time'],
                       )

    def append_to_netcdf(self):
        """Append the current atmospheric state to the netCDF4 file specified
        in ``self.outfile``.
        """
        utils.append_timestep_netcdf(
            filename=self.outfile,
            data=self.atmosphere.merge(self.heatingrates,
                                       overwrite_vars='H2O'),
            timestamp=self.get_hours_passed(),
            )

    def run(self):
        """Run the radiative-convective equilibirum model."""
        logger.info('Start RCE model run.')

        while self.niter < self.max_iterations:
            logger.debug('Enter iteration {}.'.format(self.niter))

            # Caculate shortwave, longwave and net heatingrates.
            self.calculate_heatingrates()

            # Apply heatingrates to the temperature profile.
            T = self.atmosphere['T'].values.copy()  # save old T profile.
            self.atmosphere.adjust(
                self.heatingrates['net_htngrt'],
                self.timestep
                )

            # Calculate temperature change for convergence check.
            self.atmosphere['deltaT'] = self.atmosphere['T'] - T

            # Apply heatingrates to the the surface.
            self.surface.adjust(
                self.heatingrates['net_htngrt'].values[0],
                self.timestep
                )
            logger.debug(
                f'Surface temperature: {self.surface.temperature:.4f} K'
            )

            if self.check_if_write():
                if self.niter == 0:
                    self.create_outfile()
                self.append_to_netcdf()

            if self.is_converged():
                logger.info('Converged after %s iterations.' % self.niter)
                break
            else:
                self.niter += 1
        else:
            logger.info('Stopped after maximum number of iterations.')
