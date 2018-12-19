# -*- coding: utf-8 -*-
"""Implementation of a radiative-convective equilibrium model (RCE).
"""
import logging

import numpy as np

from konrad import utils
from konrad import netcdf
from konrad.radiation import RRTMG
from konrad.ozone import (Ozone, OzonePressure)
from konrad.humidity import FixedRH
from konrad.surface import (Surface, SurfaceHeatCapacity)
from konrad.cloud import (Cloud, ClearSky)
from konrad.convection import (Convection, HardAdjustment, RelaxedAdjustment)
from konrad.lapserate import (LapseRate, MoistLapseRate)
from konrad.upwelling import (Upwelling, NoUpwelling)

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

    def __init__(self, atmosphere, timestep='3h', max_duration='5000d',
                 outfile=None, experiment='RCE', writeevery='1d', delta=1e-4,
                 radiation=None, ozone=None, humidity=None, surface=None,
                 cloud=None, convection=None, lapserate=None, upwelling=None):
        """Set-up a radiative-convective model.

        Parameters:
            atmosphere (Atmosphere): `konrad.atmosphere.Atmosphere`.
            timestep (float or str): Model time step (per iteration).
                If float, time step shall be given in days.
                If str, a timedelta string may be given
                (see :func:`konrad.utils.parse_fraction_of_day`).
            max_duration (float or str): Maximum duration of the simulation.
                The duration is given in model time:
                    If float, maximum duration in days.
                    If str, a timedelta string
                    (see `konrad.utils.parse_fraction_of_day`).
            outfile (str): netCDF4 file to store output.
            experiment (str): Experiment description (stored in netCDF output).
            writeevery(int, float or str): Set output frequency.
                Values can be given in:
                    int: Every nth iteration
                    float: Every nth day in model time
                    str: a timedelta string
                    (see `konrad.utils.parse_fraction_of_day`).
            delta (float): Stop criterion. If the heating rate is below this
                threshold for all levels, skip further iterations. Values
                are given in K/day.
            radiation (konrad.radiation): Radiation model.
                Defaults to :class:`konrad.radiation.RRTMG`.
            ozone (konrad.ozone): Ozone model.
                Defaults to :class:`konrad.ozone.OzonePressure`.
            humidity (konrad.humidity): Humidity model.
                Defaults to :class:`konrad.humidity.FixedRH`.
            surface (konrad.surface): Surface model.
                Defaults to :class:`konrad.surface.SurfaceHeatCapacity`.
            cloud (konrad.cloud): Cloud model.
                Defaults to :class:`konrad.cloud.ClearSky`.
            convection (konrad.humidity.Convection): Convection scheme.
                Defaults to :class:`konrad.convection.HardAdjustment`.
            lapserate (konrad.lapse.LapseRate): Lapse rate handler.
                Defaults to :class:`konrad.lapserate.MoistLapseRate`.
            upwelling (konrad.upwelling.Upwelling):
                Defaults to :class:`konrad.upwelling.NoUpwelling`.
        """
        # Sub-models.
        self.atmosphere = atmosphere
        if radiation is None:
            self.radiation = RRTMG()
        else:
            self.radiation = radiation

        self.ozone = utils.return_if_type(ozone, 'ozone',
                                          Ozone, OzonePressure())

        self.humidity = FixedRH() if humidity is None else humidity
        self.surface = utils.return_if_type(surface, 'surface',
                                            Surface, SurfaceHeatCapacity())
        self.cloud = utils.return_if_type(cloud, 'cloud',
                                          Cloud,
                                          ClearSky(self.atmosphere['plev'].size)
                                          )
        self.convection = utils.return_if_type(convection, 'convection',
                                               Convection, HardAdjustment())

        self.lapserate = utils.return_if_type(lapserate, 'lapserate',
                                              LapseRate, MoistLapseRate())

        self.upwelling = utils.return_if_type(upwelling, 'upwelling',
                                              Upwelling, NoUpwelling())

        self.max_duration = utils.parse_fraction_of_day(max_duration)
        self.timestep = utils.parse_fraction_of_day(timestep)
        self.writeevery = utils.parse_fraction_of_day(writeevery)

        self.max_iterations = np.ceil(self.max_duration / self.timestep)
        self.niter = 0

        self.delta = delta
        self.deltaT = None
        self.converged = False

        self.outfile = outfile
        self.nchandler = None
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
        """Return the number of hours passed since model start.

        Returns:
            float: Hours passed since model start.
        """
        return self.niter * 24 * self.timestep

    def is_converged(self):
        """Check if the atmosphere is in radiative-convective equilibrium.

        Returns:
            bool: ``True`` if converged, else ``False``.
        """
        # TODO: Implement proper convergence criterion (e.g. include TOA).
        return np.all(np.abs(self.deltaT) < self.delta)

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

    def run(self):
        """Run the radiative-convective equilibrium model."""
        logger.info('Start RCE model run.')

        # Initialize surface pressure to be equal to lowest half-level
        # pressure. This is consistent with handling in PSrad.
        self.surface.pressure = self.atmosphere['phlev'][0]

        # Main loop to control all model iterations until maximum number is
        # reached or a given stop criterion is fulfilled.
        while self.niter < self.max_iterations:
            if self.niter % 100 == 0:
                # Write every 100th time step in loglevel INFO.
                logger.info(f'Enter iteration {self.niter}.')
            else:
                # All other iterations are only logged in DEBUG level.
                logger.debug(f'Enter iteration {self.niter}.')

            self.radiation.adjust_solar_angle(self.get_hours_passed() / 24)
            self.radiation.update_heatingrates(
                atmosphere=self.atmosphere,
                surface=self.surface,
                cloud=self.cloud,
            )

            # Apply heatingrates/fluxes to the the surface.
            self.surface.adjust(
                sw_down=self.radiation['sw_flxd'][0, 0],
                sw_up=self.radiation['sw_flxu'][0, 0],
                lw_down=self.radiation['lw_flxd'][0, 0],
                lw_up=self.radiation['lw_flxu'][0, 0],
                timestep=self.timestep,
            )

            # Save the old temperature profile. They are compared with
            # adjusted values to check if the model has converged.
            T = self.atmosphere['T'].copy()

            # Caculate critical lapse rate.
            critical_lapserate = self.lapserate(self.atmosphere)

            # Apply heatingrates to temperature profile.
            self.atmosphere['T'] += (self.radiation['net_htngrt'] *
                                     self.timestep)

            # Convective adjustment
            self.convection.stabilize(
                atmosphere=self.atmosphere,
                lapse=critical_lapserate,
                timestep=self.timestep,
                surface=self.surface,
            )

            # Upwelling induced cooling
            self.upwelling.cool(
                atmosphere=self.atmosphere,
                convection=self.convection,
                timestep=self.timestep,
            )

            # TODO: Consider implementing an Atmosphere.update_diagnostics()
            #  method to include e.g. convective top in the output.
            self.atmosphere.update_height()
            z = self.atmosphere.get('z')[0, :]
            if isinstance(self.convection, HardAdjustment) or isinstance(
                    self.convection, RelaxedAdjustment):
                self.convection.calculate_convective_top_height(z)

            # Update the ozone profile.
            self.ozone(
                atmosphere=self.atmosphere,
                convection=self.convection,
                timestep=self.timestep,
                zenith=self.radiation.current_solar_angle
            )

            # Update the humidity profile.
            self.humidity.adjust_humidity(
                atmosphere=self.atmosphere,
                convection=self.convection,
                surface=self.surface,
            )

            self.cloud.update_cloud_profile(self.atmosphere,
                                            convection=self.convection)

            # Calculate temperature change for convergence check.
            self.deltaT = (self.atmosphere['T'] - T) / self.timestep

            # Check, if the current iteration is scheduled to be written.
            if self.check_if_write():
                if self.nchandler is None:
                    self.nchandler = netcdf.NetcdfHandler(
                        filename=self.outfile, rce=self)

                self.nchandler.write()

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
