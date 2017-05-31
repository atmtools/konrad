# -*- coding: utf-8 -*-
"""Module containing classes describing different atmosphere models.
"""

__all__ = [
    'Atmosphere',
    'AtmosphereFixedVMR',
    'AtmosphereFixedRH',
    'AtmosphereConvective',
    'AtmosphereMoistConvective',
    'AtmosphereConUp',
]


import abc
import collections
import logging

import typhon
import numpy as np
from xarray import Dataset, DataArray

from conrad import constants
from conrad import utils


logger = logging.getLogger()

atmosphere_variables = [
    'T',
    'z',
    'H2O',
    'N2O',
    'O3',
    'CO2',
    'CO',
    'CH4',
]


class Atmosphere(Dataset, metaclass=abc.ABCMeta):
    """Abstract base class to define requirements for atmosphere models."""
    @abc.abstractmethod
    def adjust(self, heatingrate, timestep):
        """Adjust atmosphere according to given heatingrate."""

    @classmethod
    def from_atm_fields_compact(cls, atmfield):
        """Convert an ARTS atm_fields_compact [0] into an atmosphere.

        [0] http://arts.mi.uni-hamburg.de/docserver-trunk/variables/atm_fields_compact

        Parameters:
            atmfield (typhon.arts.types.GriddedField4): A compact set of
                atmospheric fields.
        """
        # Create a Dataset with time and pressure dimension.
        plev = atmfield.grids[1]
        phlev = utils.calculate_halflevels(plev)
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        })

        for var in atmosphere_variables:
            # Get ARTS variable name from variable description.
            arts_key = constants.variable_description[var].get('arts_name')

            # Extract profile from atm_fields_compact
            profile = typhon.arts.atm_fields_compact_get(
                [arts_key], atmfield).squeeze()

            d[var] = DataArray(profile[np.newaxis, :], dims=('time', 'plev',))

        d.append_description()  # Append variable descriptions.

        return d

    @classmethod
    def from_dict(cls, dictionary):
        """Create an atmosphere model from dictionary values.

        Parameters:
            dictionary (dict): Dictionary containing ndarrays.
        """
        # TODO: Currently working for good-natured dictionaries.
        # Consider allowing a more flexibel user interface.

        # Create a Dataset with time and pressure dimension.
        plev = dictionary['plev']
        phlev = utils.calculate_halflevels(plev)
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        })

        for var in atmosphere_variables:
            d[var] = DataArray(dictionary[var], dims=('time', 'plev',))

        d.append_description()  # Append variable descriptions.

        return d

    def append_description(self):
        """Append variable attributes to dataset."""
        for key in self.keys():
            if key in constants.variable_description:
                self[key].attrs = constants.variable_description[key]

    # TODO: This function could handle the nasty time dimension in the future.
    # Allowing to set two-dimensional variables using a 1d-array, if one
    # coordinate has the dimension one.
    def set(self, variable, value):
        """Set the values of a variable.

        Parameters:
            variable (str): Variable key.
            value (float or ndarray): Value to assign to the variable.
                If a float is given, all values are filled with it.
        """
        if isinstance(value, collections.Container):
            self[variable].values[0, :] = value
        else:
            self[variable].values.fill(value)

    def get_values(self, variable, keepdims=True):
        """Get values of a given variable.

        Parameters:
            variable (str): Variable key.
            keepdims (bool): If this is set to False, single-dimensions are
                removed. Otherwise dimensions are keppt (default).

        Returns:
            ndarray: Array containing the values assigned to the variable.
        """
        if keepdims:
            return self[variable].values
        else:
            return self[variable].values.ravel()

    @property
    def relative_humidity(self):
        """Return the relative humidity of the current atmospheric state."""
        vmr, p, T = self['H2O'], self['plev'], self['T']
        return typhon.atmosphere.relative_humidity(vmr, p, T)

    @relative_humidity.setter
    def relative_humidity(self, RH):
        """Set the water vapor mixing ratio to match given relative humidity.

        Parameters:
            RH (ndarray or float): Relative humidity.
        """
        logger.debug('Adjust VMR to preserve relative humidity.')
        self['H2O'] = typhon.atmosphere.vmr(RH, self['plev'], self['T'])
        self['H2O'][0, self['H2O'][0, :] < 3e-6] = 3e-6
        # self['H2O'][0, self['plev'] < 50e2] = 3e-6
        self.set('H2O', utils.ensure_decrease(self['H2O'][0, :]))

    def get_lapse_rates(self):
        """Calculate the temperature lapse rate at each level."""
        lapse_rate = np.diff(self['T'][0, :]) / np.diff(self['z'][0, :])
        lapse_rate = typhon.math.interpolate_halflevels(lapse_rate)
        lapse_rate = np.append(lapse_rate[0], lapse_rate)
        return np.append(lapse_rate, lapse_rate[-1])

    def find_first_unstable_layer(self, critical_lapse_rate=-0.0065,
                                  pmin=10e2):
        lapse_rate = self.get_lapse_rates()
        for n in range(len(lapse_rate) - 1, 1, -1):
            if lapse_rate[n] < critical_lapse_rate and self['plev'][n] > pmin:
                return n

    # def convective_adjustment(self, critical_lapse_rate=-0.0065, pmin=10e2):
    #     i = self.find_first_unstable_layer(
    #             critical_lapse_rate=critical_lapse_rate,
    #             pmin=pmin)

    #     if i is not None:
    #         self['T'][0, :i] = (self['T'][0, i] + critical_lapse_rate
    #                             * (self['z'][0, :i] - self['z'][0, i]))


class AtmosphereFixedVMR(Atmosphere):
    """Atmosphere model with fixed volume mixing ratio."""
    def adjust(self, heatingrate, timestep):
        """Adjust the temperature.

        Adjust the atmospheric temperature profile by simply adding the given
        heatingrates.

        Parameters:
            heatingrate (float or ndarray):
                Heatingrate (already scaled with timestep) [K].
        """
        self['T'] += heatingrate * timestep


class AtmosphereFixedRH(Atmosphere):
    """Atmosphere model with fixed relative humidity.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio.
    """
    def adjust(self, heatingrate, timestep):
        """Adjust the temperature and preserve relative humidity.

        Parameters:
            heatingrate (float or ndarray):
                Heatingrate (already scaled with timestep) [K].
        """
        # Store initial relative humidty profile.
        RH = self.relative_humidity

        self['T'] += heatingrate * timestep  # adjust temperature profile.

        self.relative_humidity = RH  # reset original RH profile.


class AtmosphereConvective(Atmosphere):
    """Atmosphere model with preserved RH and fixed temperature lapse rate.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio. In addition, a simple
    convection parameterization is used.

    Implementation of Sally's convection scheme.
    """
    def convective_top(self, lapse=0.0065):
        """Find the top of the convective layer, so that energy is conserved.

        Parameters:
            lapse (float): lapse rate value to adjust to
        """
        p = self['plev']
        z = self['z'][0, :]
        T_rad = self['T'][0, :]
        density = typhon.physics.density(p, T_rad)
        Cp = constants.isobaric_mass_heat_capacity

        # Fixed lapse rate case
        start_index = self.find_first_unstable_layer()
        if start_index is None:
            return

        for a in range(start_index, len(z)):
            term2 = np.trapz((density*Cp*(T_rad+z*lapse))[:a], z[:a])
            term1 = (T_rad[a]+z[a]*lapse)*np.trapz((density*Cp)[:a], z[:a])
            if (term1 - term2) > 0:
                break
        return a

    def convective_adjustment(self, ctop, lapse=0.0065):
        """Apply the convective adjustment.

        Parameters:
            ctop (float): array index,
                the top level for the convective adjustment
            lapse (float):
                adjust to this lapse rate value
        """
        z = self['z'][0, :]
        T_rad = self['T'][0, :]
        T_con = T_rad.copy()
        for level in range(0, ctop):
            T_con[level] = T_rad[ctop] - (z[level]-z[ctop])*lapse

        self['T'].values = T_con.values[np.newaxis, :]

    def adjust(self, heatingrates, timestep):
        RH = self.relative_humidity

        self['T'] += heatingrates * timestep
        con_top = self.convective_top()
        self.convective_adjustment(con_top)

        self.relative_humidity = RH  # adjust VMR to preserve RH profile.


class AtmosphereConUp(AtmosphereConvective):
    """Atmosphere model with preserved RH and fixed temperature lapse rate,
    that includes a cooling term due to upwelling in the statosphere.
    """
    def upwelling_adjustment(self, ctop, timestep, w=0.0005):
        """Stratospheric cooling term parameterizing large-scale upwelling.

        Parameters:
            ctop (float): array index,
                the bottom level for the upwelling
                at and above this level, the upwelling is constant with height
            w (float): upwelling velocity
        """
        Cp = constants.isobaric_mass_heat_capacity
        g = constants.earth_standard_gravity

        actuallapse = self.get_lapse_rates()

        Q = -w * (-actuallapse + g / Cp)
        Q[:ctop] = 0

        self['T'] += Q * timestep

    def adjust(self, heatingrates, timestep, w=5):
        RH = self.relative_humidity

        self['T'] += heatingrates * timestep

        con_top = self.convective_top()
        self.upwelling_adjustment(con_top, timestep, w)

        self.convective_adjustment(con_top)

        self.relative_humidity = RH  # reset original RH profile.


class AtmosphereMoistConvective(Atmosphere):
    """Atmosphere model with preserved RH and a temperature and humidity
    -dependent lapse rate.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio. In addition, a convection
    parameterization is used, which sets the lapse rate to the moist adiabat.
    """
    def convective_adjustment(self, lapse):
        p = self['plev']
        z = self['z'][0, :]
        T_rad = self['T'][0, :]
        density = typhon.physics.density(p, T_rad)
        Cp = constants.isobaric_mass_heat_capacity

        for a in range(10, len(z)):
            term2b = np.trapz((density*Cp*T_rad)[:a], z[:a])
            term1 = T_rad[a]*np.trapz((density*Cp)[:a], z[:a])
            inintegral = np.zeros((a,))
            for b in range(0, a):
                inintegral[b] = np.trapz(lapse[b:a], z[b:a])
            term3 = np.trapz((density*Cp)[:a]*inintegral, z[:a])
            if (term1 + term3 - term2b) > 0:
                break

        T_con = T_rad.copy()
        for level in range(0, a):
            lapse_sum = np.trapz(lapse[level:a], z[level:a])
            T_con[level] = T_rad[a] + lapse_sum

        self['T'].values = T_con.values[np.newaxis, :]

    def adjust(self, heatingrates, timestep):

        # TODO: Calculate moist lapse rates instead of hardcoding them.
        # # calculate lapse rate based on previous con-rad state.
        # g = 9.8076
        # Lv = 2501000
        # R = 287
        # eps = 0.62197
        # Cp = 1003.5
        # VMR = self['H2O'][0, :]
        # T = self['T'][0, :]
        # lapse = g*(1 + Lv*VMR/R/T)/(Cp + Lv**2*VMR*eps/R/T**2)

        # Hardcoded moist adiabatic lapse rate.
        lapse = np.array([
            0.003498,  0.003498,  0.003598,  0.003777,  0.003924,
            0.004085,  0.004514,  0.005101,  0.005637,  0.006281,
            0.006493,  0.006739, 0.007079,  0.007453,  0.007753,
            0.00808,  0.008324,  0.008585, 0.008792,  0.009,
            0.009169,  0.009321,  0.009447,  0.009537, 0.009624,
            0.009661,  0.009698,  0.00972,  0.009736,  0.00975,
            0.009753,  0.009757,  0.009759,  0.009761,  0.009762,
            0.009763,  0.009764,  0.009764,  0.009765,  0.009765,
            0.009765,  0.009765, 0.009765,  0.009766,  0.009766,
            0.009766,  0.009766,  0.009766, 0.009766,  0.009767,
            0.009767,  0.009767,  0.009767,  0.009766, 0.009766,
            0.009766,  0.009766,  0.009766,  0.009766,  0.009766,
            0.009766,  0.009766,  0.009766,  0.009766,  0.009766,
            0.009766, 0.009766,  0.009765,  0.009765,  0.009765,
            0.009765,  0.009765, 0.009765,  0.009765,  0.009765,
            0.009765,  0.009765,  0.009765, 0.009765,  0.009765,
            0.009765,  0.009765,  0.009765,  0.009765, 0.009765,
            0.009765,  0.009765,  0.009765,  0.009765,  0.009765,
            0.009765,  0.009765,  0.009765,  0.009765,  0.009765,
            0.009764,  0.009764,  0.009764,  0.009764,  0.009764,
            0.009764,  0.009764, 0.009764,  0.009764,  0.009764,
            0.009764,  0.009765,  0.009765, 0.009765,  0.009765,
            0.009765,  0.009765,  0.009764,  0.009764, 0.009764,
            0.009764,  0.009764,  0.009764,  0.009764,  0.009764,
            0.009764,  0.009764,  0.009764,  0.009764,  0.009764,
            0.009764, 0.009764,  0.009764,  0.009764,  0.009764,
            0.009763,  0.009763, 0.009763,  0.009763,  0.009763,
            0.009763,  0.009763,  0.009763, 0.009763,  0.009763,
            0.009763,  0.009763,  0.009763,  0.009763, 0.009763,
            0.009763,  0.009763,  0.009763,  0.009763,  0.009763
            ])

        RH = self.relative_humidity

        self['T'] += heatingrates * timestep
        self.convective_adjustment(lapse)

        self.relative_humidity = RH
