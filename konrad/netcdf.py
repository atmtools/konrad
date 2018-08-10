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
    def __init__(self, filename):
        self.filename = filename
        self.groups = []
        self._component_cache = set()

        self.create_file()

    def create_file(self):
        netCDF4.Dataset(self.filename, mode='w')

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

    def get_unlimited_dimension(self):
        with netCDF4.Dataset(self.filename, 'w') as root:
            for dim in root.dimensions:
                if dim.isunlimited():
                    return dim

    def append_group(self, component, groupname):
        with netCDF4.Dataset(self.filename, 'a') as root:
            group = root.groups[groupname]

            for varname, (data, dims) in component.data_vars.items():
                s = [root.dimensions[d].size
                     if root.dimensions[d].isunlimited()
                     else slice(None)
                     for d in dims]

                group.variables[varname][tuple(s)] = data

    def get_components(self, rcemodel):
        if len(self._component_cache) == 0:
            for attr in dir(rcemodel):
                if isinstance(getattr(rcemodel, attr), Component):
                    self._component_cache.add(attr)

        return sorted(self._component_cache)

    def write(self, rcemodel):
        #TODO (lkluft): Only works if `atmosphere` is first, because we need
        #  the coordinates (plev, phlev) set.
        for component in self.get_components(rcemodel):
            if component in self.groups:
                self.append_group(getattr(rcemodel, component), component)
            else:
                self.create_group(getattr(rcemodel, component), component)

