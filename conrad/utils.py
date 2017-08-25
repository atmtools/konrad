# -*- coding: utf-8 -*-
"""Common utility functions.
"""
import collections
import logging
import os

import numpy as np
import typhon
from netCDF4 import Dataset

from conrad import constants


__all__ = [
    'append_timestep_netcdf',
    'create_relative_humidity_profile',
    'ensure_decrease',
    'calculate_halflevel_pressure',
    'append_description',
    'refined_pgrid',
    'revcumsum',
    'extract_metadata',
    'radiation_budget',
    'equilibrium_sensitivity',
    'get_filepath',
    'create_pardirs',
]

logger = logging.getLogger(__name__)


def append_timestep_netcdf(filename, data, timestamp):
    """Append a timestep to an existing netCDF4 file.

    Notes:
        The variables to append to have to exist in the netCDF4 file!

    Parameters:
        filename (str): Path to the netCDF4.
        data (dict{ndarray}): Dict-like object containing the data arrays.
            The key is the variable name and the value is an ``ndarray``, a
            ``pandas.Series`` or an ``xarray.DataArray`` e.g.:
                >>> data = {'T': np.array([290, 295, 300])}

        timestamp (float): Timestamp of values appended.
    """
    # Open netCDF4 file in `append` mode.
    with Dataset(filename, 'a') as nc:
        logging.debug('Append timestep to "{}".'.format(filename))
        t = nc.dimensions['time'].size  # get index to store data.
        nc.variables['time'][t] = timestamp  # append timestamp.

        # Append data for each variable in ``data``.
        for var in data:
            # Append variable if it has a `time` dimension and is no
            # dimension itself.
            if 'time' in nc[var].dimensions and var not in nc.dimensions:
                # TODO: Find a cleaner way to handle different data dimensions.
                if 'plev' in nc[var].dimensions:
                    if hasattr(data[var], 'values'):
                        nc.variables[var][t, :] = data[var].values
                    else:
                        nc.variables[var][t, :] = data[var]
                else:
                    if hasattr(data[var], 'values'):
                        nc.variables[var][t] = data[var].values
                    else:
                        nc.variables[var][t] = data[var]


def create_relative_humidity_profile(p, rh_s=0.75):
    """Create an exponential relative humidity profile.

    Parameters:
        p (ndarray): Pressure.
        rh_s (float): Relative humidity at first pressure level.

    Returns:
        ndarray: Relative humidtiy."""
    rh = rh_s / (np.exp(1) - 1) * (np.exp(p / p[0]) - 1)
    return np.round(rh, decimals=4)


def ensure_decrease(array):
    """Ensure that a given array is decreasing.

    Parameters:
        array (ndarray): Input array.

    Returns:
        ndarray: Monotonously decreasing array.
    """
    for i in range(1, len(array)):
        if array[i] > array[i-1]:
            array[i] = array[i-1]
    return array


def calculate_halflevel_pressure(fulllevels):
    """Returns the linear interpolated halflevels for given array.

    Parameters:
        fulllevels (ndarray): Pressure at fullevels.

    Returns:
        ndarray: Coordinates at halflevel.

    """
    inter = (fulllevels[1:] + fulllevels[:-1]) / 2
    bottom = fulllevels[0] - 0.5 * (fulllevels[1] - fulllevels[0])
    top = 0
    return np.hstack((bottom, inter, top))


def append_description(dataset, description=None):
    """Append variable attributes to a given dataset.

    Parameters:
          dataset (xarray.Dataset): Dataset including variables to describe.
          description (dict): Dictionary containing variable descriptions.
            The keys are the variable keys used in the Dataset.
            The values are dictionaries themselves containing attributes
            and their names as keys, e.g.:
                desc = {'T': {'units': 'K', 'standard_name': 'temperature'}}
    """
    if description is None:
        description = constants.variable_description

    for key in dataset.keys():
        if key in description:
            dataset[key].attrs = constants.variable_description[key]


def refined_pgrid(start, stop, num=200, shift=0.5, fixpoint=0.):
    """Create a pressure grid with adjustable distribution in logspace.

    Notes:
          Wrapper for ``typhon.math.squeezable_logspace``.

    Parameters:
        start (float): The starting value of the sequence.
        stop (float): The end value of the sequence.
        num (int): Number of sample to generate (Default is 50).
        squeeze (float): Factor with which the first stepwidth is
            squeezed in logspace. Has to be between  ``(0, 2)`.
            Values smaller than one compress the gridpoints,
            while values greater than 1 strecht the spacing.
            The default is ``0.5`` (bottom heavy.)
        fixpoint (float): Relative fixpoint for squeezing the grid.
            Has to be between ``[0, 1]``. The  default is ``0`` (bottom).

    Returns:
        ndarray: Pressure grid.
    """
    grid = typhon.math.squeezable_logspace(
        start=start, stop=stop, num=num, squeeze=shift, fixpoint=fixpoint
    )

    return grid

def revcumsum(x):
    """Returns the reversed cumulative sum of an array.

    Paramters:
        x (ndarray): Array.

    Returns:
        ndarray: Reversed cumulative sum.

    Example:
        >>> revcumsum(np.array([0, 1, 2, 3, 4]))
        array([10, 10, 9, 7, 4])
    """
    return x[::-1].cumsum()[::-1]


def extract_metadata(filepath, delimiter='_'):
    """Extract meta information for simulation from filename.

    Naming convention for the function to work properly:
        /path/to/output/{atmosphere}_{experiment}_{scale}.nc

    Additional information may be appended:
        /path/to/output/{atmosphere}_{experiment}_{scale}_{extra_info}.nc

    Parameters:
        filepath (str): Path to output file.
        delimiter (str): Delimiter used to separate meat information.

    Returns:
        collections.namedtuple: Extracted information on `atmosphere`,
            `experiment`, `scale` and additional `extra` information.

    Examples:
        >>> extract_metadata('results/tropical_nlayers_100.nc')
        Fileinformation(atmosphere='tropical', experiment='nlayers',
                        scale='100', extra='')

        >>> extract_metadata('results/tropical_nlayers_100_fast.nc')
        Fileinformation(atmosphere='tropical', experiment='nlayers',
                        scale='100', extra='fast')
    """
    # Trim parentdir and extension from given path. This simplifies the
    # extraction of the wanted information in the next steps.
    filename = os.path.splitext(os.path.basename(filepath))[0]

    # Extract information on atmosphere, experiment and scale from filename.
    # Optional remaining fields are collected as "extra" information.
    atmosphere, experiment, scale, *extra = filename.split(delimiter)

    # Define namedtuple to access results more conveniently.
    nt = collections.namedtuple(
        typename='Fileinformation',
        field_names=['atmosphere', 'experiment', 'scale', 'extra'],
    )

    return nt._make((atmosphere, experiment, scale, delimiter.join(extra)))


def radiation_budget(lw_flxu, lw_flxd, sw_flxu, sw_flxd):
    """Calculate the net radiation budget.

    Notes:
          Positive values imply a net downward flux.

    Parameters:
        lw_flxu (ndarray): Longwave upward flux [W/m^2].
        lw_flxd (ndarray): Longwave downward flux [W/m^2].
        sw_flxu (ndarray): Shortwave upward flux [W/m^2].
        sw_flxd (ndarray): Shortwave downward flux [W/m^2].

    Returns:
        ndarray: Net radiation budget [W/m^2].
    """
    return ((sw_flxd + lw_flxd) - (sw_flxu + lw_flxu))


def equilibrium_sensitivity(temperature, forcing):
    """Calculate the equilibrium climate sensitivity.

    The equilibrium climate sensitivity is given by the temperature difference
    between the first and last timestep divided by the initial radiative
    forcing.

    Parameters:
          temperature (ndarray): Surface temperature [K].
          forcing (ndarray): Radiative forcing [W/m^2].
    Returns:
          float, float: Climate sensitivity [K / (W/m^2)],
            initial radiative forcing [W/m^2].

    Examples:
        >>> temperature = np.linspace(300, 304, 25)
        >>> forcing = np.linspace(3.7, 0., temperature.size)
        >>> equilibrium_sensitivity(temperature, forcing)
        (1.0810810810810809, 3.7000000000000002)
    """
    return (temperature[-1] - temperature[0]) / forcing[0], forcing[0]


def get_filepath(atmosphere='**', experiment='**', scale='**', extra='',
                 result_dir='results', create_tree=True):
    """Returns path to netCDF4 file for given model run.

    Naming convention for the filepaths:
        {result_dir}/{season}/{experiment}/{season}_{experiment}_{scale}.nc

    Additional information may be appended:
        {result_dir}/{season}/{experiment}/{season}_{experiment}_{scale}_{extra}.nc

    Parameters:
          atmosphere (str): Initial atmosphere identifier.
          experiment (str): Experiment identifier.
          scale (str): Changed factor in specific run.
          extra (str): Additional information.
          result_dir (str): Parent directory to store all results.
          create_tree (bool): If ``True``, the directory tree is created.

    Returns:
          str: Full path to netCDF4 file.
    """
    if extra == '':
        # Combine information on initial atmosphere, experiment and chosen
        # scale factor to a full netCDF filename.
        filename = f'{atmosphere}_{experiment}_{scale}.nc'
    else:
        # If additional information is given, place it right before the
        # file extension.
        filename = f'{atmosphere}_{experiment}_{scale}_{extra}.nc'

    # Join the directories, subdirectories and filename to a full path.
    fullpath = os.path.join(result_dir, atmosphere, experiment, filename)

    # Ensure that all parent directories exist.
    # This allows to directly use the returned filepath.
    if create_tree:
        # Do not create trees for glob patterns (include `*`)!
        # This prevents the creation of nasty directories like `*` or `**`.
        if '*' not in fullpath:
            create_pardirs(fullpath)

    return fullpath


def create_pardirs(path):
    """Create all directories in a given file path.

    Examples:
          The following call will ensure, that the
          directories `foo`, `bar` and `baz` exist:
          >>> create_pardirs('foo/bar/baz/test.txt')

    Parameters:
          path (str): Filepath.
    """
    # Create the full given directory tree, regardless if it already exists.
    os.makedirs(os.path.dirname(path), exist_ok=True)
