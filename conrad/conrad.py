# -*- coding: utf-8 -*-
"""Implementation of a radiative-convective equilibrium model.

This module defines the class `ConRad` only.
"""
import logging

import numpy as np

from . import utils
from . import plots


logger = logging.getLogger()

__all__ = [
    'ConRad',
]


# Define long names and units for netCDF4 export of variables.
_netcdf_vars = {
    'CH4': ('Methane', 'VMR'),
    'CO': ('Carbon monoxide', 'VMR'),
    'CO2': ('Carbon dioxide', 'VMR'),
    'N2O': ('Nitrogen', 'VMR'),
    'O3': ('Ozone', 'VMR'),
    'Q': ('Water vapor', 'VMR'),
    'RH': ('Relative humidity', '1'),
    'T': ('Temperature', 'K'),
}


class ConRad():
    """Implementation of a radiative-convective equilibrium model.

    Examples:
        Create an object to setup and run a simulation:
        >>> c = ConRad(data=pands.DataFrame(...))
        >>> c.run()
    """
    def __init__(self, atmosphere, surface, outfile=None, dt=1, delta=0.01,
                 max_iterations=5000):
        """Set-up a radiative-convective model.

        Parameters:
            atmosphere (Atmosphere): `conrad.atmosphere.Atmosphere`.
            surface (Surface): An surface object inherited from
                `conrad.surface.Surface`.
            outfile (str): netCDF4 file to store output.
            dt (float): Time step in days.
            max_iterations (int): Maximum number of iterations.
            delta (float): Stop criterion. If the heating rate is below this
                threshold for all levels, skip further iterations.
        """
        self.atmosphere = atmosphere
        self.converged = False
        self.delta = delta
        self.dt = dt
        self.heatingrates = None
        self.max_iterations = max_iterations
        self.niter = 0
        self.outfile = outfile
        self.surface = surface

        logging.info('Created ConRad object:\n{}'.format(self))

    def __repr__(self):
        # List of attributes to include in __repr__ output.
        repr_attrs = [
            'delta',
            'dt',
            'max_iterations',
            'niter',
            ]

        retstr = '{}(\n'.format(self.__class__.__name__)
        for a in repr_attrs:
            retstr += '    {}={},\n'.format(a, getattr(self, a))
        retstr += ')'

        return retstr

    def plot_sounding_p(self, variable, ax=None, **kwargs):
        return plots.atmospheric_profile_p(
                self.atmosphere.index, self.atmosphere[variable], **kwargs)

    def plot_sounding_z(self, variable, ax=None, **kwargs):
        return plots.atmospheric_profile_z(
                self.atmosphere['Z'], self.atmosphere[variable], **kwargs)

    def plot_overview_z(self, axes=None, **kwargs):
        return plots.plot_overview_z(
            self.atmosphere,
            self.heatingrates['lw_htngrt'],
            self.heatingrates['sw_htngrt'],
            axes,
            **kwargs)

    def plot_overview_p(self, axes=None, **kwargs):
        return plots.plot_overview_z(
            self.atmosphere,
            self.heatingrates['lw_htngrt'],
            self.heatingrates['sw_hrngrt'],
            axes,
            **kwargs)

    def get_datetime(self):
        """Return the timestamp for the current iteration."""
        return self.niter * 24 * self.dt

    @utils.with_psrad_symlinks
    def calculate_heatingrates(self):
        """Use PSRAD to calculate shortwave, longwave and net heatingsrates."""
        from . import psrad

        self.heatingrates = psrad.psrad_heatingrates(
                self.atmosphere,
                self.surface,
                )

    def is_converged(self):
        """Check if equilibirum is reached.

        Returns:
            bool: ``True`` if converged, else ``False``.
        """
        return np.all(np.abs(self.heatingrates['net_htngrt']) < self.delta)

    def create_outfile(self):
        """Create netCDF4 file to store simulation results."""
        self.atmosphere.to_netcdf(self.outfile,
                                  mode='w',
                                  unlimited_dims=['time'],
                                  )

    def append_to_netcdf(self):
        """Append the current atmospheric state to the netCDF4 file specified
        in `self.outfile`.
        """
        # Append variables to netCDF4 file.
        utils.append_timestep_netcdf(
            filename=self.outfile,
            data=self.atmosphere,
            timestamp=self.get_datetime(),
            )

    # The decorator prevents that the symlinks are created and removed during
    # each iteration.
    @utils.with_psrad_symlinks
    def run(self):
        """Run the radiative-convective equilibirum model."""
        logger.info('Start RCE model run.')

        if self.outfile is not None:
            self.create_outfile()

        while self.niter < self.max_iterations:
            self.niter += 1
            logger.debug('Enter iteration {}.'.format(self.niter))

            # Caculate shortwave, longwave and net heatingrates.
            self.calculate_heatingrates()

            # Apply heatingrates to temperature profile...
            self.atmosphere.adjust(self.dt * self.heatingrates['net_htngrt'])

            # and the surface.
            self.surface.adjust(
                self.dt * self.heatingrates['net_htngrt'].values[0])

            if self.outfile is not None:
                self.append_to_netcdf()

            if self.is_converged():
                logger.info('Converged after %s iterations.' % self.niter)
                break
        else:
            logger.info('Stopped after maximum number of iterations.')
