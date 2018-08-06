# -*- coding: utf-8 -*-
import logging

import typhon
import netCDF4
import numpy as np
from scipy.interpolate import interp1d
from xarray import Dataset, DataArray

from konrad import constants
from konrad import utils
from konrad.convection import (Convection, HardAdjustment)
from konrad.lapserate import (LapseRate, MoistLapseRate)
from konrad.upwelling import (Upwelling, NoUpwelling)

__all__ = [
    'Atmosphere',
]

logger = logging.getLogger(__name__)


class Atmosphere(Dataset):
    """Implementation of the atmosphere component."""
    atmosphere_variables = [
        'T',
        'H2O',
        'N2O',
        'O3',
        'O2',
        'CO2',
        'CO',
        'CH4',
        'CFC11',
        'CFC12',
        'CFC22',
        'CCl4',
    ]

    def __init__(self, convection=None, lapse=None,
                 upwelling=None, **kwargs):
        """Create an atmosphere model.

        Parameters:
            convection (konrad.humidity.Convection): Convection scheme.
                Defaults to ``konrad.convection.HardAdjustment``.
             lapse (konrad.lapse.LapseRate): Lapse rate handler.
                Defaults to ``konrad.lapserate.MoistLapseRate``.
            upwelling (konrad.upwelling.Upwelling): Upwelling model.
                Defaults to ``konrad.upwelling.NoUpwelling``.
        """
        # Initialize ``xarray.Dataset`` with given positional args and kwargs.
        super().__init__(**kwargs)

        # Check input types.
        convection = utils.return_if_type(convection, 'convection',
                                          Convection, HardAdjustment())

        lapse = utils.return_if_type(lapse, 'lapse',
                                     LapseRate, MoistLapseRate())

        upwelling = utils.return_if_type(upwelling, 'upwelling',
                                         Upwelling, NoUpwelling())

        # Set additional attributes for the Atmosphere object. They can be
        # accessed through point notation but do not need to be of type
        # ``xarrayy.DataArray``.
        self.attrs.update({
            'convection': convection,
            'lapse': lapse,
            'upwelling': upwelling,
        })

    def adjust(self, heatingrate, timestep, surface, **kwargs):
        """Adjust temperature according to given heating rates.

        Parameters:
            heatingrate (ndarray): Radiative heatingrate [K/day].
            timestep (float): Timestep width [day].
        """
        # Caculate critical lapse rate.
        lapse = self.lapse.get(self)

        # Apply heatingrates to temperature profile.
        self['T'] += heatingrate * timestep

        # Convective adjustment
        self.convection.stabilize(atmosphere=self, lapse=lapse,
                                  timestep=timestep, surface=surface)

        # Upwelling induced cooling
        self.upwelling.cool(atmosphere=self, radheat=heatingrate[0, :],
                            timestep=timestep)

        # Calculate the geopotential height field.
        self.update_height()

    def create_variable(self, name, data=None, dims=None):
        """Createa a variable entry in the dataframe."""
        if dims is None:
            dims = constants.variable_description[name].get('dims')

        if data is None:
            data = self.get_default_profile(name)

        ndim = len(dims)
        if ndim == 2 and data.ndim == 1:
                data = data[np.newaxis, :]

        self[name] = DataArray(data, dims=dims)

        self[name].attrs = constants.variable_description.get(name, {})

    def get_default_profile(self, name):
        """Return a profile with default values."""
        try:
            vmr = constants.variable_description[name]['default_vmr']
        except KeyError:
            raise Exception(f'No default specified for "{name}".')
        else:
            return vmr * np.ones(self.plev.size)

    @classmethod
    def from_atm_fields_compact(cls, atm_fields_compact, **kwargs):
        """Convert an ARTS atm_fields_compact [0] into an atmosphere.

        [0] http://arts.mi.uni-hamburg.de/docserver-trunk/variables/atm_fields_compact

        Parameters:
            atm_fields_compact (typhon.arts.types.GriddedField4):
                Compact set of atmospheric fields.
        """
        def _extract_profile(atmfield, species):
            try:
                arts_key = constants.variable_description[species]['arts_name']
            except KeyError:
                logger.warning(f'No variabel description for "{species}".')
            else:
                return atmfield.get(arts_key, keep_dims=False)

        datadict = {var: _extract_profile(atm_fields_compact, var)
                    for var in cls.atmosphere_variables}
        datadict['plev'] = atm_fields_compact.grids[1]

        return cls.from_dict(datadict, **kwargs)

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
                'XML file contains "{}". Expected "GriddedField4".'.format(
                    arts_type)
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
        #TODO: [Discussion] Do we want to read the actual half-level pressure?
        phlev = utils.phlev_from_plev(plev)
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        },
                **kwargs,
                )

        for var in cls.atmosphere_variables:
            d.create_variable(var, dictionary.get(var))

        # Calculate the geopotential height.
        d.update_height()

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
        def _return_profile(ds, var, ts):
            return (ds[var][ts, :] if 'time' in ds[var].dimensions
                    else ds[var][:])

        with netCDF4.Dataset(ncfile) as dataset:
            datadict = {var: _return_profile(dataset, var, timestep)
                        for var in cls.atmosphere_variables
                        if var in dataset.variables
                        }
            datadict['plev'] = dataset['plev'][:]

        return cls.from_dict(datadict)

    def to_atm_fields_compact(self):
        """Convert an atmosphere into an ARTS atm_fields_compact."""
        # Store all atmosphere variables including geopotential height.
        variables = self.atmosphere_variables + ['z']

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

    def to_netcdf(self, *args, **kwargs):
        """Write atmosphere contents to a netCDF file.

        Note:
             This is a wrapper simple for `xr.Dataset.to_netcdf`. The method
             converts all attributes (e.g. humidity, convection) to strings.
             This is necessary because the netCDF standard only allows
             string attributes.

        """
        def stringify(obj):
            """Return strings, integers and floats. Else return class name."""
            if isinstance(obj, (str, int, float)):
                return obj
            else:
                return obj.__class__.__name__

        attributes = self.attrs.copy()

        self.attrs = {key: stringify(val) for key, val in self.attrs.items()}

        super().to_netcdf(*args, **kwargs)

        self.attrs = attributes

    def refine_plev(self, pgrid, **kwargs):
        """Refine the pressure grid of an atmosphere object.

        Note:
              This method returns a **new** object,
              the original object is maintained!

        Parameters:
              pgrid (ndarray): New pressure grid [Pa].
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
        for variable in self.atmosphere_variables:
            # and create an interpolation function using the original data.
            f = interp1d(self['plev'].values, self[variable],
                         axis=-1, fill_value='extrapolate', **kwargs)

            # Store the interpolated new data in the data directory.
            # dims = self.default_dimensions[variable]
            # datadict[variable] = DataArray(f(pgrid), dims=dims)
            datadict[variable] = f(pgrid).ravel()

        # Create a new atmosphere object from the filled data directory.
        # This method also calculates the new phlev coordinates.
        new_atmosphere = type(self).from_dict(datadict)

        # Keep attributes of original atmosphere object.
        # This is **extremely** important because references to e.g. the
        # convection scheme or the humidity handling are stored as attributes!
        new_atmosphere.attrs = {**self.attrs}

        # Calculate the geopotential height.
        new_atmosphere.update_height()

        # Append variable descriptions to the Dataset.
        utils.append_description(new_atmosphere)

        return new_atmosphere

    def set(self, variable, value):
        """Set the values of a variable.

        Parameters:
            variable (str): Variable key.
            value (float or ndarray): Value to assign to the variable.
                If a float is given, all values are filled with it.
        """
        self[variable][:] = value

    def get_values(self, variable, default=None, keepdims=True):
        """Get values of a given variable.

        Parameters:
            variable (str): Variable key.
            keepdims (bool): If this is set to False, single-dimensions are
                removed. Otherwise dimensions are kept (default).
            default (float): Default value assigned to all pressure levels,
                if the variable is not found.

        Returns:
            ndarray: Array containing the values assigned to the variable.
        """
        try:
            values = self[variable].values
        except KeyError:
            if default is not None:
                values = default * np.ones(self['plev'].size)
            else:
                raise KeyError(f"'{variable}' not found and no default given.")

        return values if keepdims else values.ravel()

    def calculate_height(self):
        """Calculate the geopotential height."""
        g = constants.earth_standard_gravity

        plev = self['plev'].values  # Air pressure at full-levels.
        phlev = self['phlev'].values  # Air pressure at half-levels.

        # Air temperature on half levels
        T_phlev = interp1d(plev, self['T'][0, :],
                           fill_value='extrapolate')(phlev)
        # Calculate the air density from current atmospheric state.
        rho_phlev = typhon.physics.density(phlev[:-1], T_phlev[:-1])

        dp = np.hstack((np.array([plev[0] - phlev[0]]), np.diff(plev)))
        # Use the hydrostatic equation to calculate geopotential height from
        # given pressure, density and gravity.
        z = np.cumsum(-dp / (rho_phlev * g))
        return z

    def update_height(self):
        """Update the value for height."""
        z = self.calculate_height()
        # If height is already in Dataset, update its values.
        if 'z' in self.data_vars:
            self.set('z', z)
        # Otherwise create the DataArray.
        else:
            self.create_variable('z', z)

    def get_cold_point_pressure(self):
        """Find the pressure at the cold point.
        The cold point is taken at the coldest temperature below 100 Pa, to
        avoid cold temperatures high in the atmosphere (below about 10 Pa)."""
        p = self['plev'].values
        T = self['T'].values[0, :]
        cp = p[np.argmin(T[np.where(p > 100)])]
        return cp

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

    def get_subsidence_convergence_max(self, radiative_cooling, pmin=10e2):
        """Return index of maximum subsidence convergence.

        Parameters:
            radiative_cooling (ndarray): Radiative cooling rates.
                Positive values for heating, negative values for cooling!
            pmin (float): Lower pressure threshold. The cold point has to
                be below (higher pressure, lower height) that value.

        Returns:
              float: Pressure of maxixum subsidence divergence [Pa].
        """
        plev = self['plev'].values
        omega = self.get_diabatic_subsidence(radiative_cooling)
        domega = np.diff(omega) / np.diff(plev[:-1])

        # The found maximum is off by 1 due to numerical differentiation in
        # the subsidence calculation. Therefore, return the pressure above.
        max_index = np.argmax(domega[plev[:-2] > pmin]) + 1
        max_plev = plev[max_index]

        self.create_variable('diabatic_convergence_max_index',[max_index])
        self.create_variable('diabatic_convergence_max_plev', [max_plev])

        return max_plev

    def get_convective_top(self, heatingrate, lim=-0.1):
        """Find the pressure where the radiative heating has a certain value.

        Note:
            In the HardAdjustment case, for a contop temperature that is not
            dependent on the number of distribution of pressure levels, it is
            better to take a value of lim not equal or very close to zero.

        Parameters:
            heatingrate (ndarray): Radiative heating rate [K/day].
            lim (float): Threshold value [K/day].

        Returns:
            float: Pressure at height of convective top [Pa].
        """
        p = self['plev'].values[:]
        T = self['T'].values[-1, :]

        # NOTE: `np.argmax` returns the first occurence of the maximum value.
        # In this example, the index of the first `True` value,
        # corresponding to the convective top, is returned.
        contop_i = int(np.argmax(heatingrate > lim))

        # Create auxiliary arrays storing the Qr, T and p values above and
        # below the threshold value. These arrays are used as input for the
        # interpolation in the next step.
        heat_array = np.array([heatingrate[contop_i-1], heatingrate[contop_i]])
        p_array = np.array([p[contop_i-1], p[contop_i]])
        T_array = np.array([T[contop_i-1], T[contop_i]])

        # Interpolate the pressure value where the heatingrate # equals `lim`.
        contop_plev = interp1d(heat_array, p_array, fill_value='extrapolate')(lim)
        contop_T = interp1d(heat_array, T_array, fill_value='extrapolate')(lim)

        self.create_variable('convective_top_plev', [contop_plev])
        self.create_variable('convective_top_temperature', [contop_T])

        return contop_plev

    def get_coldpoint_plev(self, pmin=10e2):
        """Return the cold point pressure.

        Parameters:
            pmin (float): Minimum pressure threshold. The function does not
                return pressure values smaller than this. This prevents
                finding the upper most level, which is likely to be the
                coldest level.
        """
        T = self['T'].values[-1, :]
        plev = self['plev'].values[:]

        return plev[np.argmin(T[plev > pmin])]

    def tracegases_rcemip(self):
        """Set trace gas concentrations according to the RCE-MIP configuration.

        The volume mixing ratios are following the values for the
        RCE-MIP (Wing et al. 2017) and constant throughout the atmosphere.
        """
        concentrations = {
            'CO2': 348e-6,
            'CH4': 1650e-9,
            'N2O': 306e-9,
            'CO': 0,
            'O3': utils.ozone_profile_rcemip(self.get_values('plev')),
            'CFC11': 0,
            'CFC12': 0,
            'CFC22': 0,
            'CCl4': 0,
        }

        for gas, vmr in concentrations.items():
            self.set(gas, vmr)
