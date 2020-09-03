import operator
from collections.abc import Hashable

import numpy as np
import xarray as xr

from konrad import constants


class Component:
    """Base class for all model components.

    The class implements a light-weight "book-keeping" of all attributes and
    items that are stored in an instance. This allows components to be
    conveniently stored into netCDF files.

    Example implementation of a model component:

    >>> class FancyComponent(Component):
    ...     def __init__(self):
    ...         self.attribute = 'foo'
    ...         self['variable'] = (('dimension',), [42,])
    ...         self.coords = {'dimension': [0]}

    Usage of the implemented auxiliary variables:

    >>> component = FancyComponent()
    >>> component.attrs
    {'attribute': 'foo'}
    >>> component.data_vars
    {'variable': (('dimension',), [42])}

    """
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._attrs = {}
        instance._data_vars = {}
        instance.coords = {}

        return instance

    @classmethod
    def from_netcdf(cls, ncfile, timestep=-1, *args, **kwargs):
        """Load a model component from a netCDF file.

        Note:
            The :class:`konrad.netcdf.NetcdfHandler` converts variables
            of type `double` variables into `float` to reduce disk space.
            However, the underlying `climt` library requires input
            for :class:`konrad.radiation.RRTMG` to be of type `double`.
            Make sure to perform the respective typecasting
            when reading variables from netCDF files.

        Parameters:
            ncfile (str): Path to the netCDF file.
            timestep (int): Index of timestep to read.

        """
        raise NotImplementedError(
            f"{cls.__name__} does not support reading of netCDF files."
        )

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

        if not name.startswith('_') and name != 'coords':
            self._attrs[name] = value

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError

        try:
            return self._attrs[name]
        except KeyError as exc:
            raise AttributeError(exc)

    @property
    def attrs(self):
        """Dictionary containing all attributes."""
        return self._attrs

    def __setitem__(self, key, value):
        if type(value) is tuple:
            dims, data = value
            self._data_vars[key] = value
        else:
            data = value

        dims = self._data_vars[key][0]
        self._data_vars[key] = (dims, data)

    def __getitem__(self, key):
        if key in self._data_vars:
            return self._data_vars[key][1]
        else:
            return self.coords[key]

    @property
    def data_vars(self):
        """Dictionary containing all data variables and their dimensions."""
        return self._data_vars

    @property
    def netcdf_subgroups(self):
        """Define subgroups used when storing to netCDF file.

        Components that are used as wrappers for other components (e.g.
        :class:`konrad.humidity.FixedRH`) can define names and object
        references for the wrapped components.
        Those will be used when storing to netCDF files.

        Examples:
            >>> FixedRH.netcdf_subgroups
            {'rh_func': <CacheFromAtmosphere() object at ...>,
             'stratosphere_coupling': <ColdPointCoupling() object at ...>}

        Type: dict-like
        """
        raise AttributeError('No netCDF subgroups defined.')

    def __repr__(self):
        dims = ', '.join(f'{d}: {np.size(v)}' for d, v in self.coords.items())
        return f'<{self}({dims}) object at {id(self)}>'

    def __str__(self):
        return self.__class__.__name__

    def __hash__(self):
        # Prevent hashing by default as components are most likely mutable.
        raise TypeError(f'unhashable type: {type(self).__name__}')

    def hash_attributes(self):
        """Create a hash from all **hashable** component attributes."""
        attrs_sorted_by_key = sorted(self.attrs.items(),
                                     key=operator.itemgetter(0))

        hashable_values = tuple(item[1] for item in attrs_sorted_by_key
                                if isinstance(item[1], Hashable))

        # Include the class name to distinguish between different
        # inheriting classes using the same attributes.
        return hash((self.__class__.__name__, *hashable_values))

    def to_dataset(self):
        """Convert model component into an `xarray.Dataset`."""
        if self.coords is None:
            raise Exception(
                "Could not create `xarray.Dataset`: `self.coords` not set."
            )
        else:
            self.coords['time'] = [0]
            return xr.Dataset(
                coords=self.coords,
                data_vars=self.data_vars,
                attrs=self.attrs,
            )

    def create_variable(self, name, data=None, dims=None):
        """Create a variable in the model component.

        Parameters:
            name (str): Variable name.
            data (``np.ndarray``): Data array with a shape matching the
            variable dimensions (``dims``).
            dims (tuple[str]): Tuple of strings specifying the dimension names.

        Example:
            >>> c = Component()
            >>> arr = np.arange(5)
            >>> c.create_variable('foo', data=arr, dims=('index',))
            >>> c['foo']
            array([0, 1, 2, 3, 4])

        """
        if dims is None:
            try:
                # Try to determine default dimensions for common variables.
                dims = constants.variable_description[name].get('dims')
            except KeyError:
                # It is not possible to create variables without dimension.
                raise ValueError(
                    f'Could not determine default dimensions for "{name}". '
                    'You can provide them using the `dims` keyword.'
                )

        if len(dims) == 2 and data.ndim == 1:
            # Reshape one-dimensional arrays to match two dimensions.  This
            # convenience feature allows the user to assign time-dependent
            # profiles (e.g. ``T[time, pressure]``) using one-dimensional
            # input (``arr[plev]``).
            data = data[np.newaxis, :]

        self[name] = (dims, data)

    def set(self, variable, value):
        """Set the values of a variable.

        Parameters:
            variable (str): Variable key.
            value (float or ndarray): Value to assign to the variable.
                If a float is given, all values are filled with it.
        """
        self[variable][:] = value

    def get(self, variable, default=None, keepdims=True):
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
            values = self[variable]
        except KeyError:
            if default is not None:
                values = default
            else:
                raise KeyError(f"'{variable}' not found and no default given.")

        return values if keepdims else values.ravel()

