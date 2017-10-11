# -*- coding: utf-8 -*-
"""Common utility functions.
"""
import collections
import copy
import logging
import os

import numpy as np
import typhon
from netCDF4 import Dataset
from astropy.convolution import convolve, Box1DKernel
from scipy.interpolate import interp1d

from conrad import constants


__all__ = [
    'append_timestep_netcdf',
    'ozone_profile',
    'ozonesquash',
    'create_relative_humidity_profile',
    'create_relative_humidity_profile2',
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


def ozonesquash(o3, z, squash):
    """
    Squashes the ozone profile upwards or stretches it downwards, with no 
        change to the shape of the profile above the ozone concentration maximum
    Parameters:
        o3 (ndarray): initial ozone profile
        z (ndarray): corresponding height values
        squash: float, with 1 being no squash, 
            numbers < 1 squashing the profile towards the maximum,
            numbers > 1, stretching the profile downwards
    Returns:
        ndarray: new ozone profile
    """
    i_max_o3 = np.argmax(o3)
    
    sqz = (z - z[i_max_o3])*squash + z[i_max_o3]
    new_o3 = copy.copy(o3) #TODO: check if copy is needed
    new_o3[:i_max_o3] = np.interp(z[:i_max_o3], sqz, o3)
    return new_o3


def ozone_profile(p=None, o3_dataset=None):
    """
    Open a datafile of ozone values and interpolate them onto the pressure
        grid.
    Parameters:
        p (ndarray): Pressure.
        o3_dataset (Dataset): ozone data at 200 pressure levels.
    Returns:
        ndarray: Ozone concentration corresponding to p.
    """
    # If the ozone dataset is not passed, use the shipped one.
    if o3_dataset is None:
        # Use the current module path to construct the location of the
        # data directory.
        pkg_path = os.path.dirname(__file__)
        ncfile = os.path.join(pkg_path, 'data', 'ozone_profile.nc')
        o3_dataset = Dataset(ncfile)

    # **Read** the ozone profile into a ndarray.
    o3_data = o3_dataset['O3'][:]

    # If no pressure grid is passed...
    if p is None:
        # return the original data.
        return o3_data
    else:
        # Otherwise interpolate the values to the passed pressure grid.
        p_data = np.array(o3_dataset['plev'])
        f = interp1d(p_data, o3_data, kind='cubic', fill_value='extrapolate')
        return f(p)

def create_relative_humidity_profile(p, rh_s=0.75):
    """Create an exponential relative humidity profile.

    Parameters:
        p (ndarray): Pressure.
        rh_s (float): Relative humidity at first pressure level.

    Returns:
        ndarray: Relative humidtiy."""
    rh = rh_s / (np.exp(1) - 1) * (np.exp(p / p[0]) - 1)
    return np.round(rh, decimals=4)


def create_relative_humidity_profile2(plev, rh_surface=0.90, rh_tropo=0.35,
                                      p_tropo=150e2, c=1.7):
    """Create a realistic relative humidity profile for the tropics.

    Parameters:
        plev (ndarray): Pressure [Pa].
        rh_surface (float): Relative humidity at first pressure level.
        rh_tropo (float): Relative humidity at upper-tropospheric peak.
        c (float): Factor to control the width of the
            upper-tropospheric peak.

    Returns:
        ndarray: Relative humidity.
    """
    rh = rh_surface / (np.exp(1) - 1) * (np.exp((plev / plev[0])**1.1) - 1)

    rh += rh_tropo * np.exp(-c*(np.log(plev / p_tropo)**2))

    return rh


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
        shift (float): Factor with which the first stepwidth is
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

def max_mass_divergence(ds, maxDiv=True, i=-1):
    """ Find the level of maximum mass divergence
    Sorry Lukas, this one is not so neat. Please change it!
    
    Parameters:
        ds: Dataset from netCDF4
        i: index for Dataset timestep
        maxDiv == True: returns T and z at level of maximum divergence
        maxDiv == False: returns stability (S), omega and divergence profiles  
    """
    drylapse = 0.0098
    lapse_rate = np.diff(ds['T'][i, :]) / np.diff(ds['z'][i, :])
    lapse_rate = typhon.math.interpolate_halflevels(lapse_rate)
    gamma = lapse_rate/drylapse
    Cp = constants.Cp # isobaric specific heat of dry air [J kg-1 K-1]
    Rd = 287.058 # gas constant of dry air [J kg-1, K-1]
    
    T = ds['T'][i, 1:-1]
    S = Rd/Cp * T/(ds['plev'][1:-1]/10**2) * (1 - gamma) # static stability [K hPa-1]
    Q_r = ds['net_htngrt'][i, 1:-1] # radiative cooling [K day-1]
    omega = -Q_r/S # [hPa day-1]
    d_omega1 = np.diff(omega)
    
    d_omega = convolve(d_omega1, Box1DKernel(5)) # smooth
    
    d_P = np.diff(ds['plev'])[1:-1]/10**2
    Div = d_omega/d_P # mass divergence
    
    if not maxDiv:
        return S, omega, Div
    
    z = typhon.math.interpolate_halflevels(ds['z'][i, 1:-1])
    
    cp_z = z[np.argmin(T[np.where(z < 30000)])] # z at cold point
    
    # find maximum divergence between 5 km altitude and the cold point height
    Div_5_30 = Div[np.where(z > 5000)][np.where(z < cp_z)]
    z_5_30 = z[np.where(z > 5000)][np.where(z < cp_z)]
    z_maxDiv = z_5_30[np.argmax(Div_5_30)]
    
    T_5_30 = typhon.math.interpolate_halflevels(T)[np.where(z > 5000)][np.where(z < cp_z)]
    T_maxDiv = T_5_30[np.argmax(Div_5_30)]
    
    return T_maxDiv, z_maxDiv
