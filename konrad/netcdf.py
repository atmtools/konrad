import logging
from datetime import datetime

import netCDF4
import numpy as np

from konrad import (constants, __version__)
from konrad.component import Component


__all__ = [
    'NetcdfHandler',
]

logger = logging.getLogger(__name__)


def _move_item_to_index(list, item, index):
    list.insert(index, list.pop(list.index(item)))


def convert_unsupported_types(variable):
    """Convert variables into a netCDF-supported data type."""
    if variable is None:
        return np.nan

    if isinstance(variable, bool):
        return 1 if variable else 0

    if isinstance(variable, str):
        return np.asarray([variable])

    if hasattr(variable, 'values'):
        return variable.values

    return variable


class NetcdfHandler:
    """A netCDF file handler.

    Note:
        The netCDF handler typecasts variables from ``double`` into ``float``
        in order to save disk space.
        Developers need to convert variables back into ``double``
        when implementing ``konrad.component.Component.from_netcdf()``
        methods for model components.

    Usage:
        >>> rce = konrad.RCE(...)
        >>> nc = NetcdfHandler('output.nc', rce)  # create output file
        >>> nc.write()  # write (append) current RCE state to file

    """
    def __init__(self, filename, rce):
        self.filename = filename
        self.rce = rce

        self.udim = 'time'
        self.udim_size = 0
        self.groups = []
        self._component_cache = {}

        self.create_file()

    def create_file(self):
        with netCDF4.Dataset(self.filename, mode='w') as root:
            root.setncatts({
                'title': self.rce.experiment,
                'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'source': f'konrad {__version__}',
                'references': 'https://github.com/atmtools/konrad',
            })

        logger.debug(f'Created "{self.filename}".')

    def create_dimension(self, group, name, data):
        if name not in group.dimensions:
            group.createDimension(name, np.asarray(data).size)

        logger.debug(f'Created dimension "{name}".')

    def create_variable(self, group, name, value, dims=()):
        value = convert_unsupported_types(value)

        dtype = np.asarray(value).dtype
        variable = group.createVariable(
            varname=name,
            # Store double variables in single precision to save disk space.
            datatype="float32" if dtype == "float64" else dtype,
            dimensions=dims,
            zlib=True,
        )
        variable[:] = value

        logger.debug(f'Created variable "{name}".')

        self.append_description(variable)

    def append_description(self, variable):
        desc = constants.variable_description.get(variable.name, {})

        for attribute_name, value in desc.items():
            logger.debug(
                f'Added attribute "{attribute_name}" to "{variable.name}".')
            setattr(variable, attribute_name, value)

    def create_group(self, component, groupname):
        with netCDF4.Dataset(self.filename, 'a') as root:
            group = root.createGroup(groupname)
            group.setncattr('class', type(component).__name__)

            for attr, value in component.attrs.items():
                self.create_variable(group, attr, value)

            for name, coord in component.coords.items():
                self.create_dimension(root, name, coord)
                if name not in root.variables:
                    self.create_variable(root, name, coord, (name,))

            for varname, (dims, data) in component.data_vars.items():
                if varname not in group.variables:
                    self.create_variable(group, varname, data, dims)

            logger.debug(f'Created group "{groupname}".')

            self.groups.append(groupname)

    def append_group(self, component, groupname):
        with netCDF4.Dataset(self.filename, 'a') as root:
            group = root[groupname]

            for varname, (dims, data) in component.data_vars.items():
                if self.udim not in dims:
                    continue

                s = [self.udim_size if dim == self.udim else slice(None)
                     for dim in dims]

                group.variables[varname][tuple(s)] = data

    def expand_unlimited_dimension(self):
        with netCDF4.Dataset(self.filename, 'a') as root:
            self.udim_size = root.dimensions[self.udim].size

            desc = constants.variable_description.get(self.udim, {})
            root[self.udim][self.udim_size] = netCDF4.date2num(
                self.rce.time,
                units=desc["units"],
                calendar=desc["calendar"],
            )

    def get_components(self):
        """Return a list of non-empty non-private model components."""
        if len(self._component_cache) == 0:
            # Ensure that the atmosphere component is stored first as it holds
            # the common coordinates `plev` and `phlev`.
            self._component_cache['atmosphere'] = self.rce.atmosphere

            for attr_name in dir(self.rce):
                attr = getattr(self.rce, attr_name)
                if attr_name.startswith('_') or not isinstance(attr, Component):
                    continue  # Only cache (#1) non-private (#2) components.

                if hasattr(attr, "netcdf_subgroups"):
                    for sname, sgroup in attr.netcdf_subgroups.items():
                        self._component_cache[f"{attr_name}/{sname}"] = sgroup
                else:
                    self._component_cache[attr_name] = attr

        logger.debug(f'Components for netCDF file: "{self._component_cache}".')

        return self._component_cache

    def initialize_file(self):
        for name, component in self.get_components().items():
            self.create_group(component, name)

        with netCDF4.Dataset(self.filename, 'a') as root:
            root.variables[self.udim][:] = 0

    def append_to_file(self):
        self.expand_unlimited_dimension()
        for name, component in self.get_components().items():
            self.append_group(component, name)

    def write(self):
        """Write current state of the RCE model to the netCDF file."""
        if len(self.groups) == 0:
            self.initialize_file()
        else:
            self.append_to_file()
