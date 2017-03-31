# -*- coding: utf-8 -*-
"""Implementation of a radiative-convective equilibrium model.

This module defines the class `ConRad` only.
"""
import logging
import os

import numpy as np
from typhon import atmosphere

from . import utils
from . import plots


logger = logging.getLogger()

__all__ = [
    'ConRad',
]


# Define long names and units for netCDF4 export of variables.
_netcdf_vars = {
    # 'CH4': ('Methane', 'VMR'),
    # 'CO': ('Carbon monoxide', 'VMR'),
    # 'CO2': ('Carbon dioxide', 'VMR'),
    # 'N2O': ('Nitrogen', 'VMR'),
    # 'O3': ('Ozone', 'VMR'),
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
    def __init__(self, sounding=None, fix_rel_humidity=True, fix_surface=True,
                 outfile=None, dt=1, max_iterations=5000, delta=0.03):
        """Set-up a radiative-convective model.

        Parameters:
            sounding (pd.DataFrame): pandas DataFrame representing an
                atmospheric sounding.
            fix_rel_humidity (bool): Adjust the water vapor mixing ratio to
                keep the relative humidity constant.
            fix_stick (bool): Fix surface temperature.
            outfile (str): netCDF4 file to store output.
            dt (float): Time step in days.
            max_iterations (int): Maximum number of iterations.
            delta (float): Stop criterion. If the heating rate is below this
                threshold for all levels, skip further iterations.
        """
        self.converged = False
        self.delta = delta
        self.dt = dt
        self.outfile = outfile
        self.fix_rel_humidity = fix_rel_humidity
        self.fix_surface = fix_surface
        self.max_iterations = max_iterations
        self.niter = 0
        self.heatingrates = None
        self.sounding = sounding

        logging.info('Created ConRad object:\n{}'.format(self))

    def __repr__(self):
        # List of attributes to include in __repr__ output.
        repr_attrs = [
            'delta',
            'dt',
            'fix_rel_humidity',
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
                self.sounding.index, self.sounding[variable], **kwargs)

    def plot_sounding_z(self, variable, ax=None, **kwargs):
        return plots.atmospheric_profile_z(
                self.sounding['Z'], self.sounding[variable], **kwargs)

    def plot_overview_z(self, axes=None, **kwargs):
        return plots.plot_overview_z(
            self.sounding,
            self.heatingrates['lw_htngrt'],
            self.heatingrates['sw_htngrt'],
            axes,
            **kwargs)

    def plot_overview_p(self, axes=None, **kwargs):
        return plots.plot_overview_z(
            self.sounding,
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
            self.sounding, fix_surface=self.fix_surface)

    def adjust_temperature(self):
        """Adjust the temperature profile.

        This methods calls `self.calculate_heatingsrates` and applies the
        simulated heatingrates to the current atmospheric state.
        """
        # Caculate shortwave, longwave and net heatingrates.
        self.calculate_heatingrates()

        # Apply heatingrates to temperature profile.
        self.sounding['T'] += self.dt * self.heatingrates['net_htngrt']

    def adjust_vmr(self):
        """Adjust water vapor mixing ratio to preserve relative humidity."""
        logger.debug('Adjust VMR to preserve relative humidity.')
        self.sounding['Q'] = atmosphere.vmr(
                                    self.sounding['RH'],
                                    self.sounding['P'],
                                    self.sounding['T'])

    def adjust_relative_humidity(self):
        """Adjust relative humidity to preserve water vapor mixing ratio."""
        logger.debug('Adjust relative humidity to preserve VMR.')
        self.sounding['RH'] = atmosphere.relative_humidity(
                                    self.sounding['Q'],
                                    self.sounding['P'],
                                    self.sounding['T'])

    def is_converged(self):
        """Check if equilibirum is reached.

        Returns:
            bool: ``True`` if converged, else ``False``.
        """
        return np.all(np.abs(self.heatingrates['net_htngrt']) < self.delta)

    def to_netcdf(self):
        """Store the current atmospheric state to the netCDF4 file specified in
        `self.outfile`. If the file does not exist, create it.

        New timesteps are appended to existing files.
        """
        # TODO: Consider writing a decent export framework. Including a
        # get_netcdf_vars() function and the possibility to pass such a dict to
        # the writing method.

        # TODO: Currently files are not overwritten when already existing.
        # Consider chaning this behaviour.
        if not os.path.isfile(self.outfile):
            # If the output netCDF4 file does not exist, create it.
            utils.create_netcdf(
                filename=self.outfile,
                pressure=self.sounding['P'],
                description='Radiative-convective equilibrium simulation.',
                variable_description=_netcdf_vars,
                )

        # Export all variables porperly specified with longname and unit.
        export_vars = {v: self.sounding[v].values for v in _netcdf_vars}

        # Append variables to netCDF4 file.
        utils.append_timestep_netcdf(
            filename=self.outfile,
            data=export_vars,
            timestamp=self.get_datetime(),
            )

    # The decorator prevents that the symlinks are created and removed during
    # each iteration.
    @utils.with_psrad_symlinks
    def run(self):
        """Run the radiative-convective equilibirum model."""
        logger.info('Start RCE model run.')

        # Calculate the relative humidity for given atmosphere.
        self.adjust_relative_humidity()

        while self.niter < self.max_iterations:
            self.niter += 1
            logger.debug('Enter iteration {}.'.format(self.niter))

            self.adjust_temperature()

            if self.fix_rel_humidity:
                self.adjust_vmr()
            else:
                self.adjust_relative_humidity()

            if self.outfile is not None:
                self.to_netcdf()

            if self.is_converged():
                logger.info('Converged after %s iterations.' % self.niter)
                break
        else:
            logger.info('Stopped after maximum number of iterations.')
