# -*- coding: utf-8 -*-
"""Implementation of a radiative-convective equilibrium model (RCE).
"""
import logging
from datetime import datetime

import numpy as np

from konrad import utils
from konrad.radiation import RRTMG


logger = logging.getLogger(__name__)

__all__ = [
    'RCE',
]


class RCE:
    """Interface to control the radiative-convective equilibrium simulation.

    Examples:
        Create an object to setup and run a simulation:
        >>> import konrad
        >>> rce = konrad.RCE(...)
        >>> rce.run()
    """
    def __init__(self, atmosphere, radiation=None, outfile=None, experiment='',
                 timestep=1, delta=0.01, writeevery=1, max_iterations=5000):
        """Set-up a radiative-convective model.

        Parameters:
            atmosphere (Atmosphere): `konrad.atmosphere.Atmosphere`.
            outfile (str): netCDF4 file to store output.
            experiment (str): Experiment description (stored in netCDF).
            timestep (float): Iteration time step in days.
            delta (float): Stop criterion. If the heating rate is below this
                threshold for all levels, skip further iterations.
            writeevery(int or float): Set frequency in which to write output.
                int: Every nth timestep is written.
                float: Every nth day is written.
            max_iterations (int): Maximum number of iterations.
        """
        # Sub-models.
        self.atmosphere = atmosphere
        if radiation is None:
            self.radiation = RRTMG()
        else:
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
        self.experiment = experiment

        logging.info('Created Konrad object:\n{}'.format(self))

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

    # TODO: Consider implementing netCDF writing in a cleaner way. Currently
    # variables from different Datasets are hard to distinguish. Maybe
    # dive into the group mechanism in netCDF.
    def create_outfile(self):
        """Create netCDF4 file to store simulation results."""
        data = self.atmosphere.merge(self.heatingrates, overwrite_vars='H2O')
        data.merge(self.atmosphere.surface, inplace=True)

        # Add experiment and date information to newly created netCDF file.
        data.attrs.update(experiment=self.experiment)
        data.attrs.update(date=datetime.now().strftime("%Y-%m-%d %H:%M"))

        # Not all Radiation classes provide an `solar_constant` attribute.
        # For thos who do (e.g. `RRTMG`) store the value in the netCDF file.
        if hasattr(self.radiation, 'solar_constant'):
            data.attrs.update(solar_constant=self.radiation.solar_constant)

        # The `Atmosphere.to_netcdf()` function is overloaded and able to
        # handle attributes in a proper way (saving the object's class name).
        data.to_netcdf(self.outfile, mode='w', unlimited_dims=['time'])

        logger.info(f'Created "{self.outfile}".')

    def append_to_netcdf(self):
        """Append the current atmospheric state to the netCDF4 file specified
        in ``self.outfile``.
        """
        data = self.atmosphere.merge(self.heatingrates, overwrite_vars='H2O')
        data.merge(self.atmosphere.surface, inplace=True)

        utils.append_timestep_netcdf(
            filename=self.outfile,
            data=data,
            timestamp=self.get_hours_passed(),
            )

    def run(self):
        """Run the radiative-convective equilibrium model."""
        logger.info('Start RCE model run.')

        # Initialize surface pressure to be equal to lowest half-level
        # pressure. This is consistent with handling in PSrad.
        self.atmosphere.surface['pressure'] = self.atmosphere['phlev'][0]

        # Main loop to control all model iterations until maximum number is
        # reached or a given stop criterion is fulfilled.
        while self.niter < self.max_iterations:
            if self.niter % 100 == 0:
                # Write every 100th time step in loglevel INFO.
                logger.info(f'Enter iteration {self.niter}.')
            else:
                # All other iterations are only logged in DEBUG level.
                logger.debug(f'Enter iteration {self.niter}.')

            # Adjust the solar angle according to current time.
            self.radiation.adjust_solar_angle(self.get_hours_passed() / 24)

            # Caculate shortwave, longwave and net heatingrates.
            # Afterwards, they are accesible throug ``self.heatingrates``.
            self.calculate_heatingrates()

            # Apply heatingrates/fluxes to the the surface.
            self.atmosphere.surface.adjust(
                sw_down=self.heatingrates['sw_flxd'].values[0, 0],
                sw_up=self.heatingrates['sw_flxu'].values[0, 0],
                lw_down=self.heatingrates['lw_flxd'].values[0, 0],
                lw_up=self.heatingrates['lw_flxu'].values[0, 0],
                timestep=self.timestep,
            )

            # Save the old temperature profile. They are compared with
            # adjusted values to check if the model has converged.
            T = self.atmosphere['T'].values.copy()

            # Apply heatingrates to the temperature profile.
            self.atmosphere.adjust(
                self.heatingrates['net_htngrt'],
                self.timestep,
                surface=self.atmosphere.surface,
                )

            # Calculate temperature change for convergence check.
            self.atmosphere['deltaT'] = self.atmosphere['T'] - T

            # Check, if the current iteration is scheduled to be written.
            if self.check_if_write():
                # If we are in the first iteration, a new is created...
                if self.niter == 0:
                    self.create_outfile()
                # ... otherwise we just append.
                else:
                    self.append_to_netcdf()

            # Check if the model run has converged to an equilibrium state.
            if self.is_converged():
                # If the model is converged, skip further iterations. Success!
                logger.info(f'Converged after {self.niter} iterations.')
                break
            # Otherweise increase the iteration count and go on.
            else:
                self.niter += 1
        else:
            logger.info('Stopped after maximum number of iterations.')
