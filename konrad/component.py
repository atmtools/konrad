import numpy as np
import xarray as xr

from konrad import constants


__all__ = [
    'Component',
]


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

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

        if not name.startswith('_') and name != 'coords':
            self._attrs[name] = value

    def __getattr__(self, name):
        return self._attrs[name]

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

    def __repr__(self):
        dims = ', '.join(f'{d}: {np.size(v)}' for d, v in self.coords.items())
        return f'<{self.__class__.__name__}({dims}) object at {id(self)} >'

    @property
    def netcdf_nelem(self):
        """Total number of netCDF elements (attributes and data variables."""
        return len(self.data_vars) + len(self.attrs)

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
        """Create a variable entry in the dataframe."""
        if dims is None:
            dims = constants.variable_description[name].get('dims')

        if data is None:
            data = self.get_default_profile(name)

        ndim = len(dims)
        if ndim == 2 and data.ndim == 1:
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

