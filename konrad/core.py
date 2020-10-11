"""Implementation of a radiative-convective equilibrium model (RCE). """
import datetime
import logging

import numpy as np

from konrad import constants
from konrad import utils
from konrad import netcdf
from konrad.radiation import RRTMG
from konrad.ozone import (Ozone, OzonePressure)
from konrad.humidity import FixedRH
from konrad.surface import (Surface, SlabOcean, FixedTemperature)
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

    def __init__(
        self,
        atmosphere,
        timestep='6h',
        max_duration='500d',
        outfile=None,
        experiment='RCE',
        writeevery='24h',
        delta=0.0,
        radiation=None,
        ozone=None,
        humidity=None,
        surface=None,
        cloud=None,
        convection=None,
        lapserate=None,
        upwelling=None,
        diurnal_cycle=False,
        co2_adjustment_timescale=np.nan,
        logevery=None,
        timestep_adjuster=None,
    ):
        """Set-up a radiative-convective model.

        Parameters:

            atmosphere: :py:class:`konrad.atmosphere.Atmosphere`.

            timestep (float, str or timedelta): Model time step

                * If float, time step in days.
                * If str, a timedelta string (see :func:`konrad.utils.parse_fraction_of_day`).
                * A `timedelta` object is directly used as timestep.

            max_duration (float, str or timedelta): Maximum duration.
                The duration is given in model time

                * If float, maximum duration in days.
                * If str, a timedelta string (see :func:`konrad.utils.parse_fraction_of_day`).
                * A `timedelta` object is directly used as timestep.

            outfile (str): netCDF4 file to store output.

            experiment (str): Experiment description (stored in netCDF output).

            writeevery (float, str or timedelta): Set output frequency.

                * float: Every nth day in model time
                * str: a timedelta string (see :func:`konrad.utils.parse_fraction_of_day`).
                * A `timedelta` object is directly used as timestep.
                * Note: Setting a value of `"0h"` will write after every iteration.

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
                Defaults to :class:`konrad.surface.SlabOcean`.

            cloud (konrad.cloud): Cloud model.
                Defaults to :class:`konrad.cloud.ClearSky`.

            convection (konrad.convection): Convection scheme.
                Defaults to :class:`konrad.convection.HardAdjustment`.

            lapserate (konrad.lapserate): Lapse rate handler.
                Defaults to :class:`konrad.lapserate.MoistLapseRate`.

            upwelling (konrad.upwelling): Upwelling model.
                Defaults to :class:`konrad.upwelling.NoUpwelling`.

            diurnal_cycle (bool): Toggle diurnal cycle of solar angle.

            co2_adjustment_timescale (int/float): Adjust CO2 concentrations
                towards an equilibrium state following Romps 2020.
                To be used with :class:`konrad.surface.FixedTemperature`.
                Recommended value is 7 (1 week).
                Defaults to no CO2 adjustment, with `np.nan`.

            logevery (int): Log the model progress at every nth iteration.
                Default is no logging.

                This keyword only affects the frequency with which the
                log messages are generated. You have to enable the logging
                by using the :py:mod:`logging` standard library or
                the `konrad.enable_logging()` convenience function.

            timestep_adjuster (callable): A callable object, that takes the
                current timestep and the temperature change as input
                to calculate a new timestep (`f(timestep, deltaT)`).

                Default (`None`) is keeping the given timestep fixed.
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
                                            Surface, SlabOcean())
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

        self.diurnal_cycle = diurnal_cycle

        self.timestep = utils.parse_fraction_of_day(timestep)
        self.time = datetime.datetime(1, 1, 1)  # Start model run at 0001/1/1
        self.max_duration = utils.parse_fraction_of_day(max_duration)
        self.niter = 0

        self.writeevery = utils.parse_fraction_of_day(writeevery)
        self.last_written = self.time

        self.logevery = logevery

        self.delta = delta
        self.deltaT = None
        self.converged = False

        self.outfile = outfile
        self.nchandler = None
        self.experiment = experiment

        self.co2_adjustment_timescale = co2_adjustment_timescale
        if not np.isnan(co2_adjustment_timescale) and not isinstance(
                surface, FixedTemperature):
            raise TypeError(
                "Runs with adjusting CO2 concentration "
                "require a fixed surface temperature."
                )

        self.timestep_adjuster = timestep_adjuster

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
        return self.runtime.total_seconds() / 3_600

    @property
    def runtime(self):
        """Timedelta representing time since model start."""
        return self.time - datetime.datetime(1, 1, 1)

    @property
    def timestep_days(self):
        return self.timestep.total_seconds() / constants.seconds_in_a_day

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

        if ((self.time - self.last_written) >= self.writeevery
           or self.time == datetime.datetime(1, 1, 1)):
            self.last_written = self.time
            return True
        else:
            return False

    def run(self):
        """Run the radiative-convective equilibrium model."""
        logger.info('Start RCE model run.')

        # Initialize surface pressure to be equal to lowest half-level
        # pressure. This is consistent with handling in PSrad.
        self.surface.pressure = self.atmosphere['phlev'][0]

        # Main loop to control all model iterations until maximum number is
        # reached or a given stop criterion is fulfilled.
        while self.runtime <= self.max_duration:
            if self.logevery is not None and self.niter % self.logevery == 0:
                # Write every 100th time step in loglevel INFO.
                logger.info(f'Enter iteration {self.niter}.')
                if self.timestep_adjuster is not None:
                    logger.info(f'Model timestep: {self.timestep}.')

            if self.diurnal_cycle:
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
                timestep=self.timestep_days,
            )

            if not np.isnan(self.co2_adjustment_timescale):
                # adjust CO2 concentrations to find a equilibrium state using
                # equation 8 of Romps 2020
                n0 = self.surface.heat_sink
                A = 5.35
                tau = self.co2_adjustment_timescale
                self.atmosphere['CO2'] += self.timestep_days * (
                    n0 - self.radiation['toa'][0]) / (A * tau
                                                      ) * self.atmosphere['CO2']

            # Save the old temperature profile. They are compared with
            # adjusted values to check if the model has converged.
            T = self.atmosphere['T'].copy()

            # Caculate critical lapse rate.
            critical_lapserate = self.lapserate(self.atmosphere)

            # Apply heatingrates to temperature profile.
            self.atmosphere['T'] += (self.radiation['net_htngrt'] *
                                     self.timestep_days)

            # Convective adjustment
            self.convection.stabilize(
                atmosphere=self.atmosphere,
                lapse=critical_lapserate,
                timestep=self.timestep_days,
                surface=self.surface,
            )

            # Upwelling induced cooling
            self.upwelling.cool(
                atmosphere=self.atmosphere,
                convection=self.convection,
                timestep=self.timestep_days,
            )

            # TODO: Consider implementing an Atmosphere.update_diagnostics()
            #  method to include e.g. convective top in the output.
            self.atmosphere.update_height()
            z = self.atmosphere.get('z')[0, :]
            if isinstance(self.convection, HardAdjustment) or isinstance(
                    self.convection, RelaxedAdjustment):
                self.convection.update_convective_top_height(z)

            # Update the ozone profile.
            self.ozone(
                atmosphere=self.atmosphere,
                convection=self.convection,
                timestep=self.timestep_days,
                upwelling=self.upwelling,
                zenith=self.radiation.current_solar_angle
            )

            # Update the humidity profile.
            self.humidity.adjust_humidity(
                atmosphere=self.atmosphere,
                convection=self.convection,
                surface=self.surface,
            )

            self.cloud.update_cloud_profile(
                atmosphere=self.atmosphere,
                convection=self.convection,
                radiation=self.radiation,
            )

            # Calculate temperature change for convergence check.
            self.deltaT = (self.atmosphere['T'] - T) / self.timestep_days

            if self.timestep_adjuster is not None:
                self.timestep = self.timestep_adjuster(
                    self.timestep, self.deltaT * self.timestep_days
                )

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
            # Otherwise increase the iteration count and go on.
            else:
                self.niter += 1
                self.time += self.timestep
        else:
            logger.info('Stop. Reached maximum runtime.')


class TimestepAdjuster:
    """Callable object to adjust the model timestep."""
    def __init__(
        self,
        increment=None,
        timestep_min=None,
        timestep_max=None,
        lower=0.05,
        upper=0.5,
    ):
        """Initialize a timestep adjuster.

        Parameters:
            increment (datetime.timedelta): Timestep increment.
            timestep_min (datetime.timedelta): Minimum timestep.
            timestep_max (datetime.timedelta): Maximum timestep.
            lower (float): Lower threshold for temperature change.
                If the absolute temperature change on each level
                deceeds this value, the timestep is increased.
            upper (float): Upper threshold for temperature change.
                If the absolute temperature change on each level
                exceeds this value, the timestep is decreased.
        """
        if increment is None:
            increment = datetime.timedelta(hours=1)

        if timestep_min is None:
            timestep_min = datetime.timedelta(hours=2)

        if timestep_max is None:
            timestep_max = datetime.timedelta(days=1, hours=12)

        self.increment = increment
        self.timestep_min = timestep_min
        self.timestep_max = timestep_max
        self.lower = lower
        self.upper = upper

    def __call__(self, timestep, deltaT):
        # Calculate the maximum absolute temperature change per day.
        absmax = np.abs(deltaT).max()

        if absmax < self.lower and timestep < self.timestep_max:
            timestep += self.increment
        elif absmax > self.upper and timestep:
            timestep -= 2 * self.increment

        if timestep < self.timestep_min:
            timestep = self.timestep_min

        return timestep
