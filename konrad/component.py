import numpy as np
import xarray as xr


__all__ = [
    'Component',
]


class Component:
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
        return self._data_vars

    def __repr__(self):
        dims = ', '.join(f'{d}: {np.size(v)}' for d, v in self.coords.items())
        return f'<{self.__class__.__name__}({dims}) object at {id(self)} >'

    def to_dataset(self):
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
