# -*- coding: utf-8 -*-
import abc
import collections
import logging

import typhon
import netCDF4
import numpy as np
from scipy.interpolate import interp1d
from xarray import Dataset, DataArray

from conrad import constants
from conrad import utils


__all__ = [
    'Atmosphere',
]

logger = logging.getLogger()

atmosphere_variables = [
    'T',
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
    def adjust(self, heatingrate, timestep, **kwargs):
        """Adjust atmosphere according to given heatingrate."""

    @classmethod
    def from_atm_fields_compact(cls, atmfield, **kwargs):
        """Convert an ARTS atm_fields_compact [0] into an atmosphere.

        [0] http://arts.mi.uni-hamburg.de/docserver-trunk/variables/atm_fields_compact

        Parameters:
            atmfield (typhon.arts.types.GriddedField4): A compact set of
                atmospheric fields.
        """
        # Create a Dataset with time and pressure dimension.
        plev = atmfield.grids[1]
        phlev = utils.calculate_halflevel_pressure(plev)
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        },
                **kwargs,
                )

        for var in atmosphere_variables:
            # Get ARTS variable name from variable description.
            arts_key = constants.variable_description[var].get('arts_name')

            # Extract profile from atm_fields_compact
            profile = typhon.arts.atm_fields_compact_get(
                [arts_key], atmfield).squeeze()

            d[var] = DataArray(profile[np.newaxis, :], dims=('time', 'plev',))

        # Calculate the geopotential height.
        d.calculate_height()

        # Append variable descriptions to the Dataset.
        utils.append_description(d)

        return d

    @classmethod
    def from_xml(cls, xmlfile, **kwargs):
        """Read atmosphere from XML file containing an ARTS atm_fields_compact.

        Parameters:
            xmlfile (str): Path to XML file.
        """
        # Read the content of given XML file.
        griddedfield = typhon.arts.xml.load(xmlfile)

        # Check if the XML file contains an atm_fields_compact (GriddedField4).
        arts_type = typhon.arts.utils.get_arts_typename(griddedfield)
        if arts_type != 'GriddedField4':
            raise TypeError(
                f'XML file does not contain "GriddedField4" but "{arts_type}".'
            )

        return cls.from_atm_fields_compact(griddedfield, **kwargs)

    @classmethod
    def from_dict(cls, dictionary, **kwargs):
        """Create an atmosphere model from dictionary values.

        Parameters:
            dictionary (dict): Dictionary containing ndarrays.
        """
        # TODO: Currently working for good-natured dictionaries.
        # Consider allowing a more flexibel user interface.

        # Create a Dataset with time and pressure dimension.
        plev = dictionary['plev']
        phlev = utils.calculate_halflevel_pressure(plev)
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        },
                **kwargs,
                )

        for var in atmosphere_variables:
            d[var] = DataArray(dictionary[var], dims=('time', 'plev',))

        # Calculate the geopotential height.
        d.calculate_height()

        # Append variable descriptions to the Dataset.
        utils.append_description(d)

        return d

    @classmethod
    def from_netcdf(cls, ncfile, timestep=-1, **kwargs):
        """Create an atmosphere model from a netCDF file.

        Parameters:
            ncfile (str): Path to netCDF file.
            timestep (int): Timestep to read (default is last timestep).
        """
        data = netCDF4.Dataset(ncfile).variables

        # Create a Dataset with time and pressure dimension.
        plev = data['plev'][:]
        phlev = utils.calculate_halflevel_pressure(plev)
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        },
                **kwargs,
                )

        for var in atmosphere_variables:
            d[var] = DataArray(
                data=data[var][[timestep], :],
                dims=('time', 'plev',)
            )

        # Calculate the geopotential height.
        d.calculate_height()

        # Append variable descriptions to the Dataset.
        utils.append_description(d)

        return d

    def to_atm_fields_compact(self):
        """Convert an atmosphere into an ARTS atm_fields_compact."""
        # Store all atmosphere variables including geopotential height.
        variables = atmosphere_variables + ['z']

        # Get ARTS variable name from variable description.
        species = [constants.variable_description[var].get('arts_name')
                   for var in variables]

        # Create a GriddedField4.
        atmfield = typhon.arts.types.GriddedField4()

        # Set grids and their names.
        atmfield.gridnames = ['Species', 'Pressure', 'Longitude', 'Latitude']
        atmfield.grids = [
            species, self['plev'].values, np.array([]), np.array([])
        ]

        # The profiles have to be passed in "stacked" form, as an ndarray of
        # dimensions [species, pressure, lat, lon].
        atmfield.data = np.vstack(
            [self[var].values.reshape(1, self['plev'].size, 1, 1)
             for var in variables]
        )
        atmfield.dataname = 'Data'

        # Perform a consistency check of the passed grids and data tensor.
        atmfield.check_dimension()

        return atmfield

    def refine_plev(self, pgrid, axis=1, **kwargs):
        """Refine the pressure grid of an atmosphere object.

        Note:
              This method returns a **new** object,
              the original object is maintained!

        Parameters:
              pgrid (ndarray): New pressure grid [Pa].
              axis (int): Index of pressure axis (should be 1).
                This keyword is only there for possible changes in future.
            **kwargs: Additional keyword arguments are collected
                and passed to :func:`scipy.interpolate.interp1d`

        Returns:
              Atmosphere: A **new** atmosphere object.
        """
        # Initialize an empty directory to fill it with interpolated data.
        # The dictionary is later used to create a new object using the
        # Atmosphere.from_dict() classmethod. This allows to circumvent the
        # fixed dimension size in xarray.DataArrays.
        datadict = dict()

        datadict['plev'] = pgrid  # Store new pressure grid.
        # Loop over all atmospheric variables...
        for variable in atmosphere_variables:
            # and create an interpolation function using the original data.
            f = interp1d(self['plev'].values, self[variable],
                         axis=axis, fill_value='extrapolate', **kwargs)

            # Store the interpolated new data in the data directory.
            datadict[variable] = DataArray(f(pgrid), dims=('time', 'plev'))

        # Create a new atmosphere object from the filled data directory.
        new_atmosphere = type(self).from_dict(datadict)

        # Calculate the geopotential height.
        new_atmosphere.calculate_height()

        # Append variable descriptions to the Dataset.
        utils.append_description(new_atmosphere)

        return new_atmosphere

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

    def calculate_height(self):
        """Calculate the geopotential height."""
        g = constants.earth_standard_gravity

        plev = self['plev'].values  # Air pressure at full-levels.
        phlev = self['phlev'].values  # Air pressure at half-levels.
        T = self['T'].values  # Air temperature at full-levels.

        # Calculate the air density from current atmospheric state.
        rho = typhon.physics.density(plev, T)

        # Use the hydrostatic equation to calculate geopotential height from
        # given pressure, density and gravity.
        z = np.cumsum(-np.diff(phlev) / (rho * g))

        # If height is already in Dataset, update its values.
        if 'z' in self:
            self['z'].values[0, :] = np.cumsum(-np.diff(phlev) / (rho * g))
        # Otherwise create the DataArray.
        else:
            self['z'] = DataArray(z[np.newaxis, :], dims=('time', 'plev'))

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
        self['H2O'].values = typhon.atmosphere.vmr(RH, self['plev'], self['T'])

    def get_lapse_rates(self):
        """Calculate the temperature lapse rate at each level."""
        lapse_rate = np.diff(self['T'][0, :]) / np.diff(self['z'][0, :])
        lapse_rate = typhon.math.interpolate_halflevels(lapse_rate)
        lapse_rate = np.append(lapse_rate[0], lapse_rate)
        return np.append(lapse_rate, lapse_rate[-1])

    def get_potential_temperature(self, p0=1000e2):
        """Calculate the potential temperature.

        .. math::
            \theta = T \cdot \left(\frac{p_0}{P}\right)^\frac{2}{7}

        Parameters:
              p0 (float): Pressure at reference level [Pa].

        Returns:
              ndarray: Potential temperature [K].
        """
        # Get view on temperature and pressure arrays.
        T = self['T'].values[0, :]
        p = self['plev'].values

        # Calculate the potential temperature.
        return T * (p0 / p) ** (2 / 7)

    def get_static_stability(self):
        """Calculate the static stability.

        .. math::
            \sigma = - \frac{T}{\Theta} \frac{\partial\Theta}{\partial p}

        Returns:
              ndarray: Static stability [K/Pa].
        """
        # Get view on temperature and pressure arrays.
        t = self['T'].values[0, :]
        p = self['plev'].values

        # Calculate potential temperature and its vertical derivative.
        theta = self.get_potential_temperature()
        dtheta = np.diff(theta) / np.diff(p)

        return -(t / theta)[:-1] * dtheta

    def get_diabatic_subsidence(self, radiative_cooling):
        """Calculate the diabatic subsidence.

        Parameters:
              radiative_cooling (ndarray): Radiative cooling rates.
                Positive values for heating, negative values for cooling!

        Returns:
            ndarray: Diabatic subsidence [Pa/day].
        """
        sigma = self.get_static_stability()

        return -radiative_cooling[:-1] / sigma

    def get_subsidence_convergence_max_index(self, radiative_cooling,
                                             pmin=10e2):
        """Return index of maximum subsidence convergence.

        Parameters:
            radiative_cooling (ndarray): Radiative cooling rates.
                Positive values for heating, negative values for cooling!
            pmin (float): Lower pressure threshold. The cold point has to
                be below (higher pressure, lower height) that value.

        Returns:
              int: Layer index.
        """
        plev = self['plev'].values
        omega = self.get_diabatic_subsidence(radiative_cooling)
        domega = np.diff(omega) / np.diff(plev[:-1])

        return np.argmax(domega[plev[:-2] > pmin]) + 1

    def adjust_relative_humidity(self, heatingrates, rh_surface=0.8,
                                 rh_tropo=0.3):
        """Adjust the relative humidity according to the vertical structure."""
        # TODO: Humidity values should be attributes of Atmosphere objects.

        # Find the level (index) of maximum diabatic subsidence convergence.
        # This level is associated with a second peak in the relative humidity.
        scm = self.get_subsidence_convergence_max_index(heatingrates[0, :])
        self['convergence_max'] = DataArray([scm], dims=('time',))

        # Determine relative humidity profile with second maximum at the
        # level of maximum subsidence convergence.
        plev = self['plev'].values
        rh = utils.create_relative_humidity_profile(
            plev=plev,
            rh_surface=rh_surface,
            rh_tropo=rh_tropo,
            p_tropo=plev[scm],
        )

        # Overwrite the current humidity profile with the new values.
        self.relative_humidity = rh

    @property
    def cold_point_index(self, pmin=1e2):
        """Return the pressure index of the cold point tropopause.

        Parameters:
              pmin (float): Lower pressure threshold. The cold point has to
              be below (higher pressure, lower height) that value.

        Returns:
            int: Layer index.
        """
        return int(np.argmin(self['T'][:, self['plev'] > pmin]))

    def apply_H2O_limits(self, vmr_max=1.):
        """Adjust water vapor VMR values to follow physical limitations.

        Parameters:
            vmr_max (float): Maximum limit for water vapor VMR.
        """
        # Keep water vapor VMR values above the cold point tropopause constant.
        i = self.cold_point_index
        self['H2O'].values[0, i:] = self['H2O'][0, i]

        # NOTE: This has currently no effect, as the vmr_max is set to 1.
        # Limit the water vapor mixing ratios to a given threshold.
        too_high = self['H2O'].values > vmr_max
        self['H2O'].values[too_high] = vmr_max
