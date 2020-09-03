import logging

import typhon
import netCDF4
import numpy as np
from scipy.interpolate import interp1d
from copy import copy

from konrad import constants
from konrad import utils
from konrad.component import Component

__all__ = [
    'Atmosphere',
]

logger = logging.getLogger(__name__)


class Atmosphere(Component):
    """Atmosphere component.

    Attributes:
        atmosphere_variables (list[str]): Atmospheric variables defined by the
            ``Atmosphere`` component.
        pmin (float): Minimum pressure used as threshold between upper and
            lower atmosphere [Pa]. Methods like ``get_cold_point_index`` or
            ``get_triple_point_index`` are looking for levels with higher
            pressure (closer to the surface) only.
    """
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
    pmin = 10e2

    def __init__(self, phlev):
        """Initialise atmosphere component.

        Parameters:
            phlev (``np.ndarray``): Atmospheric pressure at half-levels
              (surface to top) [Pa].
        """
        super().__init__()

        if not utils.is_decreasing(phlev):
            raise ValueError(
                "The atmospheric pressure grid has to be monotonically decreasing."
            )

        plev = utils.plev_from_phlev(phlev)

        self.coords = {
            'time': np.array([]),  # time dimension
            'plev': plev,  # pressure at full-levels
            'phlev': phlev,  # pressure at half-levels
        }

        for varname in self.atmosphere_variables:
            self.create_variable(varname, np.zeros_like(plev))

        # TODO: Combine with ``tracegases_rcemip``?
        self.create_variable(
            name='T',
            data=utils.standard_atmosphere(plev, coordinates='pressure'),
        )
        self.update_height()

        self.tracegases_rcemip()

    @classmethod
    def from_atm_fields_compact(cls, atm_fields_compact):
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

        return cls.from_dict(datadict)

    @classmethod
    def from_xml(cls, xmlfile):
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
    def from_dict(cls, dictionary):
        """Create an atmosphere model from dictionary values.

        Parameters:
            dictionary (dict): Dictionary containing ndarrays.
        """
        # TODO: Currently working for good-natured dictionaries.
        #  Consider a more flexible user interface.

        # Create a Dataset with time and pressure dimension.
        d = cls(phlev=dictionary['phlev'])

        for var in cls.atmosphere_variables:
            val = dictionary.get(var)
            if val is not None:
                # Prevent variables, that are not stored in the netCDF file,
                # to be overwritten with ``None``.
                d.create_variable(var, val)

        # Calculate the geopotential height.
        d.update_height()

        return d

    @classmethod
    def from_netcdf(cls, ncfile, timestep=-1):
        """Create an atmosphere model from a netCDF file.

        Parameters:
            ncfile (str): Path to netCDF file.
            timestep (int): Timestep to read (default is last timestep).
        """

        def _return_profile(ds, var, ts):
            return ds[var][ts, :] if 'time' in ds[var].dimensions else ds[var][:]

        with netCDF4.Dataset(ncfile) as root:
            if 'atmosphere' in root.groups:
                dataset = root['atmosphere']
            else:
                dataset = root

            datadict = {
                var: np.array(_return_profile(dataset, var, timestep), dtype="float64")
                for var in cls.atmosphere_variables if var in dataset.variables
            }
            datadict['phlev'] = np.array(root['phlev'][:], dtype="float64")

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
            species, self['phlev'], np.array([]), np.array([])
        ]

        # The profiles have to be passed in "stacked" form, as an ndarray of
        # dimensions [species, pressure, lat, lon].
        profiles = []
        for var in variables:
            f = interp1d(
                np.log(self["plev"]),
                self[var],
                kind="cubic",
                fill_value="extrapolate",
            )
            profiles.append(
                    f(np.log(self["phlev"])).reshape(1, self['phlev'].size, 1, 1)
            )
        atmfield.data = np.vstack(profiles)
        atmfield.dataname = 'Data'

        # Perform a consistency check of the passed grids and data tensor.
        atmfield.check_dimension()

        return atmfield

    def hash_attributes(self):
        """Create hash based on some basic characteristics"""
        return hash((
            self['plev'].min(),  # Pressure at top of the atmosphere
            self['plev'].max(),  # Surface pressure
            self['plev'].size,  # Number of pressure layers
            np.round(self['CO2'][0] / 1e-6),  # CO2 ppmv
            np.round(self['T'][-1, 0], 3),  # Surface temperature
        ))

    def refine_plev(self, phlev, **kwargs):
        """Refine the pressure grid of an atmosphere object.

        Note:
              This method returns a **new** object,
              the original object is maintained!

        Parameters:
            phlev (ndarray): New half-level-pressure grid [Pa].
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

        # Store new pressure grid.
        datadict['phlev'] = phlev
        plev = utils.plev_from_phlev(phlev)

        # Loop over all atmospheric variables...
        for variable in self.atmosphere_variables:
            # and create an interpolation function using the original data.
            f = interp1d(self['plev'], self[variable],
                         axis=-1, fill_value='extrapolate', **kwargs)

            # Store the interpolated new data in the data directory.
            datadict[variable] = f(plev).ravel()

        # Create a new atmosphere object from the filled data directory.
        new_atmosphere = type(self).from_dict(datadict)

        # Keep attributes of original atmosphere object.
        # This is **extremely** important because references to e.g. the
        # convection scheme or the humidity handling are stored as attributes!
        new_atmosphere.attrs.update({**self.attrs})

        # Calculate the geopotential height.
        new_atmosphere.update_height()

        return new_atmosphere

    def copy(self):
        """Create a copy of the atmosphere.

        Returns:
            konrad.atmosphere: copy of the atmosphere
        """
        datadict = dict()
        datadict['phlev'] = copy(self['phlev'])  # Copy pressure grid.

        # Create copies (and not references) of all atmospheric variables.
        for variable in self.atmosphere_variables:
            datadict[variable] = copy(self[variable]).ravel()

        # Create a new atmosphere object from the filled data directory.
        new_atmosphere = type(self).from_dict(datadict)

        return new_atmosphere

    def calculate_height(self):
        """Calculate the geopotential height."""
        g = constants.earth_standard_gravity

        plev = self['plev']  # Air pressure at full-levels.
        phlev = self['phlev']  # Air pressure at half-levels.
        T = self['T']  # Air temperature at full-levels.

        rho = typhon.physics.density(plev, T)
        dp = np.hstack((np.array([plev[0] - phlev[0]]), np.diff(plev)))

        # Use the hydrostatic equation to calculate geopotential height from
        # given pressure, density and gravity.
        return np.cumsum(-dp / (rho * g))

    def update_height(self):
        """Update the value for height."""
        z = self.calculate_height()
        # If height is already in Dataset, update its values.
        if 'z' in self.data_vars:
            self.set('z', z)
        # Otherwise create the DataArray.
        else:
            self.create_variable('z', z)

    def get_cold_point_index(self):
        """Return the model level index at the cold point.

        Returns:
            int: Model level index at the cold point.
        """
        plev = self['plev'][:]
        T = self['T'][-1, :]

        return np.argmin(T[plev > self.pmin])

    def get_cold_point_plev(self, interpolate=False):
        """Return the cold point pressure.

        Paramteres:
            interpolate (bool): If `False` return the pressure grid value of
                the actual coldest point. If `True` perform a quadratic fit
                to retrieve a smoother estimate of the cold point pressure.

        Returns:
            float: Pressure at the cold point [Pa].
        """
        if interpolate:
            # Use the single coldest level between troposphere and stratosphere
            # as starting point.
            plog = np.log(self["plev"])
            idx = self.get_cold_point_index()

            # Select a symmetric region [in ln(p)] around that level.
            mask = np.logical_and(
                plog > plog[idx] - 0.25,
                plog <= plog[idx] + 0.25,
            )

            # Fit a quadratic polynomial to the temperature profile
            # around the cold point: f(x) = a * x**2 + b*x + c
            popt = np.polyfit(
                plog[mask],
                self["T"][-1, mask],
                deg=2,
            )

            # Use a and b to determine the minmum of f(x) anayltically:
            # f'(x) = 0
            # 2 * a * x + b = 0
            # x = -b / (2 * a)
            return np.exp(-popt[1] / (2 * (popt[0])))
        else:
            # Return the single coldest point on the actual pressure grid.
            return self['plev'][self.get_cold_point_index()]

    def get_triple_point_index(self):
        """Return the model level index at the triple point.

        The triple point is taken at the temperature closest to 0 C.

        Returns:
            int: Model level index at the triple point.
        """
        plev = self['plev']
        T = self['T'][0, :]

        return np.argmin(np.abs(T[np.where(plev > self.pmin)] - 273.15))

    def get_triple_point_plev(self):
        """
        Return the pressure at the triple point.

        The triple point is taken at the temperature closest to 0 C.

        Returns:
            float: Pressure at the triple point [Pa].
        """
        return self['plev'][self.get_triple_point_index()]

    def get_lapse_rates(self):
        """Calculate the temperature lapse rate at each level."""
        return np.gradient(self['T'][0, :], self['z'][0, :])

    def get_potential_temperature(self, p0=1000e2):
        r"""Calculate the potential temperature.

        .. math::
            \theta = T \cdot \left(\frac{p_0}{P}\right)^\frac{2}{7}

        Parameters:
              p0 (float): Pressure at reference level [Pa].

        Returns:
              ndarray: Potential temperature [K].
        """
        # Get view on temperature and pressure arrays.
        T = self['T'][0, :]
        p = self['plev']

        # Calculate the potential temperature.
        return T * (p0 / p) ** (2 / 7)

    def get_static_stability(self):
        r"""Calculate the static stability.

        .. math::
            \sigma = - \frac{T}{\Theta} \frac{\partial\Theta}{\partial p}

        Returns:
              ndarray: Static stability [K/Pa].
        """
        # Get view on temperature and pressure arrays.
        t = self['T'][0, :]
        p = self['plev']

        # Calculate potential temperature and its vertical derivative.
        theta = self.get_potential_temperature()
        dtheta = np.gradient(theta, p)

        return -(t / theta) * dtheta

    def get_diabatic_subsidence(self, radiative_cooling):
        """Calculate the diabatic subsidence.

        Parameters:
              radiative_cooling (ndarray): Radiative cooling rates.
                Positive values for heating, negative values for cooling!

        Returns:
            ndarray: Diabatic subsidence [Pa/day].
        """
        sigma = self.get_static_stability()

        return -radiative_cooling / sigma

    def get_subsidence_convergence_max_index(self, radiative_cooling):
        """Return index of maximum subsidence convergence.

        Parameters:
            radiative_cooling (ndarray): Radiative cooling rates.
                Positive values for heating, negative values for cooling!

        Returns:
              int: Level index of maximum subsidence divergence.
        """
        plev = self['plev']
        omega = self.get_diabatic_subsidence(radiative_cooling)
        domega = np.gradient(omega, plev)

        # The subsidence divergence is the result of several consecutive
        # numerical derivatives. Therefore, it can be noisy which makes it hard
        # to define a distinct maximum. We use the center of mass in
        # ln(p)-space to prevent this problem.
        weights = domega.clip(min=0.0)
        _m = plev > self.pmin
        ctop = np.exp(
            np.sum(weights[_m] * np.log(plev[_m])) / np.sum(weights[_m])
        )

        # Map the calculated pressure to the closest value in the p-grid.
        max_index = np.abs(plev - ctop).argmin()

        self.create_variable('diabatic_convergence_max_index', [max_index])

        return max_index

    def get_subsidence_convergence_max_plev(self, radiative_cooling):
        """Return pressure of maximum subsidence convergence.

        Parameters:
            radiative_cooling (ndarray): Radiative cooling rates.
                Positive values for heating, negative values for cooling!

        Returns:
              float: Pressure of maximum subsidence divergence [Pa].
        """
        max_idx = self.get_subsidence_convergence_max_index(radiative_cooling)
        max_plev = self['plev'][max_idx]

        self.create_variable('diabatic_convergence_max_plev', [max_plev])

        return max_plev

    def get_heat_capacity(self):
        r"""Calculate specific heat capacity at constant pressure of moist air

        .. math::
            c_p = X \cdot (c_{p,v} - c_{p,d}) + c_{p,d}

        Returns:
            ndarray: Heat capacity [J/K/kg].
        """
        cpd = constants.isobaric_mass_heat_capacity_dry_air
        cpv = constants.isobaric_mass_heat_capacity_water_vapor
        x = self['H2O'][-1]

        return x * (cpv - cpd) + cpd

    def tracegases_rcemip(self):
        """Set trace gas concentrations according to the RCE-MIP configuration.

        The volume mixing ratios are following the values for the
        RCE-MIP (Wing et al. 2017) and constant throughout the atmosphere.
        """
        self.update_height()

        concentrations = {
            'H2O': utils.humidity_profile_rcemip(self.get('z')),
            'CO2': 348e-6,
            'CH4': 1650e-9,
            'N2O': 306e-9,
            'CO': 0,
            'O3': utils.ozone_profile_rcemip(self.get('plev')),
            'CFC11': 0,
            'CFC12': 0,
            'CFC22': 0,
            'CCl4': 0,
        }

        for gas, vmr in concentrations.items():
            self.set(gas, vmr)
