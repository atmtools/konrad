from datetime import datetime

import netCDF4
import numpy as np

from konrad import constants
from konrad.component import Component


__all__ = [
    'NetcdfHandler',
]


def convert_unsupported_types(variable):
    if variable is None:
        return np.nan

    if isinstance(variable, bool):
        return 1 if variable else 0

    return variable

class NetcdfHandler:
    def __init__(self, filename, rce):
        self.filename = filename
        self.rce = rce

        self.udim = 'time'
        self.udim_size = 0
        self.groups = []
        self._component_cache = set()

        self.create_file()

    def create_file(self):
        with netCDF4.Dataset(self.filename, mode='w') as root:
            root.setncattr('experiment', self.rce.experiment)
            root.setncattr(
                'created', datetime.now().strftime("%Y-%m-%d %H:%M")
            )

    def append_description(self, variable):
        desc = constants.variable_description.get(variable.name, {})

        for attribute_name, value in desc.items():
            setattr(variable, attribute_name, value)

    def create_variable(self, group, name, value, dims=()):
        value = convert_unsupported_types(value)

        variable = group.createVariable(
            varname=name,
            datatype=np.asarray(value).dtype,
            dimensions=dims,
        )
        variable[:] = value

        self.append_description(variable)

    def create_dimension(self, group, name, data):
        if name not in group.dimensions:
            group.createDimension(name, np.asarray(data).size)

    def create_group(self, component, groupname):
        with netCDF4.Dataset(self.filename, 'a') as root:
            group = root.createGroup(groupname)

            for attr, value in component.attrs.items():
                self.create_variable(group, attr, value)

            for name, coord in component.coords.items():
                self.create_dimension(root, name, coord)
                if name not in root.variables:
                    self.create_variable(root, name, coord, (name,))

            for varname, (dims, data) in component.data_vars.items():
                if varname not in group.variables:
                    self.create_variable(group, varname, data, dims)

            self.groups.append(groupname)

    def create(self):
        for component in self.get_components():
            self.create_group(getattr(self.rce, component), component)

        with netCDF4.Dataset(self.filename, 'a') as root:
            root.variables['time'][:] = 0

    def expand_time_dimension(self):
        with netCDF4.Dataset(self.filename, 'a') as root:
            self.udim_size = root.dimensions[self.udim].size

            root[self.udim][self.udim_size] = self.rce.get_hours_passed()

    def append_group(self, component, groupname):
        with netCDF4.Dataset(self.filename, 'a') as root:
            group = root.groups[groupname]

            for varname, (dims, data) in component.data_vars.items():
                if self.udim not in dims:
                    continue

                s = [self.udim_size if dim == self.udim else slice(None)
                     for dim in dims]

                group.variables[varname][tuple(s)] = data

    def append(self):
        self.expand_time_dimension()
        for component in self.get_components():
            self.append_group(getattr(self.rce, component), component)

    def get_components(self):
        if len(self._component_cache) == 0:
            for attr in dir(self.rce):
                if (not attr.startswith('_') and isinstance(getattr(self.rce, attr), Component)):
                    self._component_cache.add(attr)

        return sorted(self._component_cache)

    def write(self):
        #TODO (lkluft): Only works if `atmosphere` is first, because we need
        #  the coordinates (plev, phlev) set.
        if len(self.groups) == 0:
            self.create()
        else:
            self.append()

