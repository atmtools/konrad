"""This module integrates volcanic aerosols into konrad.
They can be used either in the RCE simulations or simply for radiative flux or heating
rate calculations.
The default setting is excluding volcanic aerosols.


Create an instance of an aerosol class, *e.g.* a :py:class:`VolcanoAerosol`:

    >>> import konrad
    >>> numlevels=200
    >>> plev, phlev = konrad.utils.get_pressure_grids(1000e2, 1,numlevels)
    >>> atmosphere = konrad.atmosphere.Atmosphere(plev)
    >>> aerosol = konrad.aerosol.VolcanoAerosol(
    >>>     atmosphere=..., aerosol_level_shift=..., file_ext_lw=...,
    >>>     file_ext_sw=..., file_g_sw=..., file_omega_sw=..., aerosol_type=...,
    >>>     include_sw_forcing=.., include_lw_forcing=..., include_scattering=...,
    >>>     include_absorption=...)

**In an RCE simulation**

    >>> rce = konrad.RCE(atmosphere=..., aerosol=aerosol)
    >>> rce.run()

**Calculating radiative fluxes or heating rates**

    >>> rrtmg.calc_radiation(atmosphere=..., surface=...,aerosol=aerosol)

More input files for radiative properties of aerosols are available at #TODO
"""

import os
import abc
import xarray as xr
import scipy as sc
import numpy as np
import typhon.physics as ty

from konrad.cloud import get_aerosol_waveband_data_array


class Aerosol(metaclass=abc.ABCMeta):
    """Base class to define abstract methods for all aerosol handlers.
    The default aerosol type is "no_aerosol", i.e. no interaction with
    radiation.

    Aerosol is characterized by four radiative properties:
        - Extinction in the LW spectrum (extinction_lw)
        - Extinction in the SW spectrum (extinction_sw)
        - Asymmetry factor (g)
        - Single scattering albedo (omega)
    """

    def __init__(
        self,
        numlevels,
        file_ext_lw=None,
        file_ext_sw=None,
        file_g_sw=None,
        file_omega_sw=None,
        aerosol_type="no_aerosol",
        aerosol_level_shift=0,
        include_sw_forcing=True,
        include_lw_forcing=True,
        include_scattering=True,
        include_absorption=True,
    ):
        """
        Create an aerosol layer.

        Parameters:
            numlevels (int): Number of levels in the atmosphere

            file_ext_lw, file_ext_sw, file_g_sw, file_omega_sw (str):
                paths to the files with information on LW extinction,
                SW extinction, asymmetry factor, single scattering albedo

            aerosol_type (str): choose between:
                * :code:`no_aerosol`
                    no volcanic aerosols
                * :code:`all_aerosol_properties`
                    volcanic aerosols, described by LW (ext) and SW (ext, g,
                                                                     ssa)

            aerosol_level_shift (float): shift of the aerosol layer relative
                to the original aerosol layer (units in km)

            include_sw_forcing, include_lw_forcing (Bool): switch on/off
                interaction of radiation with SW / LW aerosol radiative
                properties.

            include_scattering, include_absorption (Bool): include SW
                scattering / absorption component.
                (only for "all_aerosol_properties")
        """
        self.numlevels = numlevels
        self.file_ext_lw = file_ext_lw
        self.file_ext_sw = file_ext_sw
        self.file_g_sw = file_g_sw
        self.file_omega_sw = file_omega_sw
        self._aerosol_type = aerosol_type
        self.include_sw_forcing = include_sw_forcing
        self.include_lw_forcing = include_lw_forcing
        self.aerosol_level_shift = aerosol_level_shift
        self.include_scattering = include_scattering
        self.include_absorption = include_absorption
        self.optical_thickness_due_to_aerosol_sw = get_aerosol_waveband_data_array(
            0, units="dimensionless", numlevels=numlevels, sw=True
        )  # called ext_sw in files
        self.single_scattering_albedo_aerosol_sw = get_aerosol_waveband_data_array(
            0, units="dimensionless", numlevels=numlevels, sw=True
        )  # called omega_sw in files
        self.asymmetry_factor_aerosol_sw = get_aerosol_waveband_data_array(
            0, units="dimensionless", numlevels=numlevels, sw=True
        )  # called g_sw in files
        self.optical_thickness_due_to_aerosol_lw = get_aerosol_waveband_data_array(
            0, units="dimensionless", numlevels=numlevels, sw=False
        )  # called ext_lw in files

    def calculate_height_levels(self, atmosphere):
        return


class VolcanoAerosol(Aerosol):
    """
    Class for stratospheric aerosol.
    """

    def __init__(
        self,
        atmosphere,
        file_ext_lw=None,
        file_ext_sw=None,
        file_g_sw=None,
        file_omega_sw=None,
        aerosol_level_shift=0,
        include_sw_forcing=True,
        include_lw_forcing=True,
        include_scattering=True,
        include_absorption=True,
    ):
        """
        Initialize a stratospheric aerosol layer.

        Parameters:
            atmosphere (konrad.atmosphere): Corresponding atmosphere instance of the RCE

            file_ext_lw, file_ext_sw, file_g_sw, file_omega_sw (str or None):
                Path to the files with LW extinction, SW extinction,
                SW asymmetry factor, SW single scattering albedo. If None, use the
                default files (EVA, averaged between 23 N and 23 S, aerosol optical
                properties from September 1991 after a 10 Tg eruption in June 1991).
                Further files can be created based on the default files. There are also
                further examples available at [#TODO: include path to Zenodo here].

            aerosol_level_shift (float): shift of the aerosol layer relative
                to the original aerosol layer (units in km)

            include_sw_forcing, include_lw_forcing (Bool): switch on/off
                interaction of radiation with SW / LW aerosol radiative
                properties.

            include_scattering, include_absorption (Bool): include SW
                scattering / absorption component.
                (only for "all_aerosol_properties")
        """
        super().__init__(
            numlevels=np.size(atmosphere.coords["plev"]),
            aerosol_type="all_aerosol_properties",
            file_ext_lw=file_ext_lw,
            file_ext_sw=file_ext_sw,
            file_g_sw=file_g_sw,
            file_omega_sw=file_omega_sw,
            aerosol_level_shift=aerosol_level_shift,
            include_sw_forcing=include_sw_forcing,
            include_lw_forcing=include_lw_forcing,
            include_scattering=include_scattering,
            include_absorption=include_absorption,
        )

        # if no files specified, load default files
        for attr, attr_name, defaultfile in zip(
            [file_ext_lw, file_ext_sw, file_g_sw, file_omega_sw],
            ["file_ext_lw", "file_ext_sw", "file_g_sw", "file_omega_sw"],
            [
                os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    "Aerosol",
                    "EVA_10Tg_ext_lw_1991-09.nc",
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    "Aerosol",
                    "EVA_10Tg_ext_sw_1991-09.nc",
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    "Aerosol",
                    "EVA_10Tg_g_sw_1991-09.nc",
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    "Aerosol",
                    "EVA_10Tg_omega_sw_1991-09.nc",
                ),
            ],
        ):
            if attr is None:
                self.__setattr__(attr_name, defaultfile)

        # the input data has to be scaled to fit to model levels
        # for compatability with rrtmg input format
        heights = self.calculate_height_levels(atmosphere)
        scaling = np.gradient(heights)

        # switch: toggle inclusion of this radiative property
        # rrtmg_key: RRTMG accesses the radiative property by this name
        # file: where to read it from
        # bands: name of the band coordinate in the file
        # filevarname: name of the variable to read from the file
        for switch, rrtmg_key, file, bands, filevarname in zip(
            [
                self.include_lw_forcing,
                self.include_sw_forcing,
                self.include_sw_forcing,
                self.include_sw_forcing,
            ],
            [
                "optical_thickness_due_to_aerosol_lw",
                "optical_thickness_due_to_aerosol_sw",
                "asymmetry_factor_aerosol_sw",
                "single_scattering_albedo_aerosol_sw",
            ],
            [self.file_ext_lw, self.file_ext_sw, self.file_g_sw, self.file_omega_sw],
            ["terrestrial_bands", "solar_bands", "solar_bands", "solar_bands"],
            ["ext_earth", "ext_sun", "g_sun", "omega_sun"],
        ):
            if switch:
                # read ext, g, omega files if the respective switches are set True
                with xr.open_dataset(file) as dataset:
                    # shift in height
                    if self.aerosol_level_shift:
                        self.aerosol_level_shift_array = (
                            self.aerosol_level_shift
                            * np.ones(np.shape(dataset.altitude[:]))
                        )
                        dataset = dataset.assign_coords(
                            {
                                "altitude": dataset.altitude
                                + self.aerosol_level_shift_array
                            }
                        )
                    # set values of dataset to values read from the file, interpolated
                    # to pressure levels
                    getattr(self, rrtmg_key).values = sc.interpolate.interp1d(
                        dataset.altitude,
                        dataset[filevarname],
                        bounds_error=False,
                        fill_value=0,
                    )(heights)
                # scale extinction by scaling factor
                if rrtmg_key in [
                    "optical_thickness_due_to_aerosol_lw",
                    "optical_thickness_due_to_aerosol_sw",
                ]:
                    getattr(self, rrtmg_key).values *= scaling

            if not self.include_scattering:
                # only absorption: omega'=0, ext'=ext*(1-omega)"""
                try:
                    a = get_aerosol_waveband_data_array(
                        1, units="dimensionless", numlevels=self.numlevels, sw=True
                    )
                    result = (
                        np.multiply(
                            self.optical_thickness_due_to_aerosol_sw,
                            np.subtract(a, self.single_scattering_albedo_aerosol_sw),
                        ),
                    )

                    self.optical_thickness_due_to_aerosol_sw = (
                        get_aerosol_waveband_data_array(
                            result[0].values.T,
                            units="dimensionless",
                            numlevels=self.numlevels,
                            sw=True,
                        )
                    )
                    self.single_scattering_albedo_aerosol_sw = (
                        get_aerosol_waveband_data_array(
                            0,
                            units="dimensionless",
                            numlevels=self.numlevels,
                            sw=True,
                        )
                    )
                    if not self.include_absorption:
                        raise ValueError(
                            "Scattering and absorption cannot both be deactivated."
                        )
                except ValueError:
                    exit("Please choose valid input data.")

            if not self.include_absorption:
                # only scattering: omega'=1, ext'=ext*omega"""
                try:
                    result = np.multiply(
                        self.optical_thickness_due_to_aerosol_sw,
                        self.single_scattering_albedo_aerosol_sw,
                    )
                    self.optical_thickness_due_to_aerosol_sw = (
                        get_aerosol_waveband_data_array(
                            result[0].values.T,
                            units="dimensionless",
                            numlevels=self.numlevels,
                            sw=True,
                        )
                    )
                    self.single_scattering_albedo_aerosol_sw = (
                        get_aerosol_waveband_data_array(
                            1,
                            units="dimensionless",
                            numlevels=self.numlevels,
                            sw=True,
                        )
                    )

                    if not self.include_scattering:
                        raise ValueError(
                            "Scattering and absorption cannot both be deactivated."
                        )
                except ValueError:
                    exit("Please choose valid input data.")

    def calculate_height_levels(self, atmosphere):
        """Used to translate the aerosol forcing data given as a function of height
        to aerosol forcing data as a function of pressure levels
        """
        heights = ty.pressure2height(atmosphere["plev"], atmosphere["T"][0, :]) / 1000
        return heights


class NoAerosol(Aerosol):
    """
    No volcanic aerosol
    """

    def __init__(self, atmosphere):
        """
        Parameters:
            atmosphere: Pass the corresponding atmosphere instance of the RCE.
        """
        super().__init__(
            numlevels=np.size(atmosphere.coords["plev"]), aerosol_type="no_aerosol"
        )

    def calculate_height_levels(self, atmosphere):
        return
