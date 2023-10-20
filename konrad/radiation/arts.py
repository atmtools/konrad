import logging
import os
import warnings
from os.path import join, dirname, isfile

import numpy as np
import typhon as ty
from scipy.interpolate import PchipInterpolator

from konrad.utils import get_quadratic_pgrid
from konrad.atmosphere import Atmosphere
from konrad.cloud import ClearSky
from .rrtmg import RRTMG
from .common import fluxes2heating
from .radiation import Radiation


from scipy.constants import speed_of_light as c

import xarray as xr
import pyarts

from pyarts.workspace import arts_agenda

logger = logging.getLogger(__name__)


# ARTS agendas for Rayleigh scattering and Lambretian surface reflectivity
@arts_agenda
def gas_scattering_agenda__Rayleigh(ws):
    """Create an ARTS Rayleigh scattering agenda
    """
    ws.Ignore(ws.rtp_vmr)
    ws.gas_scattering_coefAirSimple()
    ws.gas_scattering_matRayleigh()


# surface scattering agenda for a lambertian surface with user defined reflectivity
@arts_agenda
def iy_surface_agenda_Lambertian(ws):
    """Create an ARTS Lambertian surface agenda
    """
    ws.iySurfaceInit()
    ws.Ignore(ws.dsurface_rmatrix_dx)
    ws.Ignore(ws.dsurface_emission_dx)
    ws.iySurfaceLambertian()
    ws.iySurfaceLambertianDirect()


class _ARTS:
    def __init__(
        self,
        ws=None,
        threads=None,
        nstreams=4,
        scale_vmr=True,
        verbosity=0,
        quadrature=False,
        quadrature_filename_lw=None,
        lookup_filename_lw=None,
        quadrature_filename_sw=None,
        lookup_filename_sw=None,
    ):
        """Initialize a wrapper for an ARTS workspace.

        Parameters:
            ws (pyarts.workspace.Workspace): An array of two ARTS workspaces - ws[0] for the longwave and ws[1] for the shortwave
            threads (int): Number of threads to use.
                Default is all available threads.
            nstreams (int): Number of viewing angles to base the radiative
                flux calculation on.
            scale_vmr (bool): Control whether dry volume mixing ratios are
                scaled with the water-vapor concentration (default is `False.`)
            verbosity (int): Control the ARTS verbosity from 0 (quiet) to 2.

            quadrature (bool): use quadrature scheme or LBL calculation via ARTS (default)
            quadrature_filename_lw (str): where to find the file that contains the wavenumbers (cm^{-1})
                and weights to compute the quadrature scheme. This file must contain at least the variable S (wavenumbers) and
                W (weights). Default is the result of a quadrature scheme trained on 72,000 frequencies x 50 profiles x 55 vertical
                levels generated with pyARTS spectroscopy with CKDMIP's eval1 atmospheric conditions, resulting in 64
                frequencies/weights, found in the data folder. The wavenumbers should be in ascending order and the weights
                should be in the corresponding order.
            lookup_filename_lw (str): filename where the lookup table corresponding to the chosen frequencies can be found. If
                None, use ARTS to generate monochromatic fluxes on the fly.
            quadrature_filename_sw (str): where to find the file that contains the wavenumbers (cm^{-1})
                and weights to compute the quadrature scheme. This file must contain at least the variable S (wavenumbers) and
                W (weights). Default is the result of a quadrature scheme trained on 62,000 frequencies x 50 profiles x 55 vertical
                levels generated with pyARTS spectroscopy with CKDMIP's eval1 atmospheric conditions, resulting in 64
                frequencies/weights, found in the data folder. The wavenumbers should be in ascending order and the weights
                should be in the corresponding order.
            lookup_filename_sw (str): filename where the lookup table corresponding to the chosen frequencies can be found. If
                None, use ARTS to generate monochromatic fluxes on the fly.


        """
        from pyarts.workspace import Workspace

        self.nstreams = nstreams
        self.scale_vmr = scale_vmr
        self._quadrature = quadrature

        self._sun_path = (
            "arts-xml-data/star/Sun/solar_spectrum_July_2008_konrad_scale.xml"
        )
        self._sun_spectrum = pyarts.xml.load(self._sun_path)
        
        if quadrature_filename_lw is None:
            # set default quadrature name, assign lookup table
            quadrature_filename_lw = join(dirname(__file__), "data", "64_quadrature_lw.h5")
            lookup_filename_lw = join(dirname(__file__), "data", "quadrature_lookup_lw.xml")
            
        if quadrature_filename_sw is None:
            quadrature_filename_sw = join(dirname(__file__), "data", "64_quadrature_sw.h5")
            lookup_filename_sw = join(dirname(__file__), "data", "quadrature_lookup_sw.xml")


        if ws is None:
            lw_ws = Workspace(verbosity=verbosity)
            sw_ws = Workspace(verbosity=verbosity)
            self._ws = [lw_ws, sw_ws]
           
        ### lw
        self._ws[0].PlanetSet(option="Earth")
        self._ws[0].AtmosphereSet1D()

        self._ws[0].water_p_eq_agendaSet()
        self._ws[0].gas_scattering_agendaSet()
        self._ws[0].iy_main_agendaSet(option="Emission")
        self._ws[0].ppath_agendaSet(option="FollowSensorLosPath")
        self._ws[0].ppath_step_agendaSet(option="GeometricPath")
        self._ws[0].iy_space_agendaSet(option="CosmicBackground")
        self._ws[0].iy_surface_agendaSet(option="UseSurfaceRtprop")

        # Non reflecting surface
        self._ws[0].VectorSetConstant(self._ws[0].surface_scalar_reflectivity, 1, 0.0)
        self._ws[0].surface_rtprop_agendaSet(
            option="Specular_NoPol_ReflFix_SurfTFromt_surface"
        )

        # Number of Stokes components to be computed
        self._ws[0].IndexSet(self._ws[0].stokes_dim, 1)

        # Set Absorption Species
        self._ws[0].abs_speciesSet(
            species=[
                "O2, O2-CIAfunCKDMT100",
                "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
                "O3",
                "CO2, CO2-CKDMT252",
                "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
                "N2O",
                "CH4",
                "CO",
            ]
        )

        ### sw
        self._ws[1].AgendaCreate("gas_scattering_agenda__Rayleigh")
        self._ws[1].AgendaCreate("iy_surface_agenda_Lambertian")

        self._ws[1].PlanetSet(option="Earth")
        self._ws[1].isotopologue_ratiosInitFromBuiltin()

        self._ws[1].iy_surface_agenda = iy_surface_agenda_Lambertian
        self._ws[1].gas_scattering_agenda = gas_scattering_agenda__Rayleigh
        self._ws[1].iy_main_agendaSet(option="Clearsky")
        self._ws[1].iy_space_agendaSet(option="CosmicBackground")
        self._ws[1].ppath_step_agendaSet(option="GeometricPath")
        self._ws[1].ppath_agendaSet(option="FollowSensorLosPath")

        self._ws[1].IndexSet(self._ws[1].stokes_dim, 1)

        # Reference ellipsoid
        self._ws[1].refellipsoidEarth(self._ws[1].refellipsoid, "Sphere")

        self._ws[1].lon_true = [0.0]
        self._ws[1].AtmosphereSet1D()

        self._ws[1].abs_speciesSet(
            species=[
                "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
                "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
                "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
                "CH4",
                "CO2, CO2-CKDMT252",
                "N2O",
                "CO",
                "O3",
                "O3-XFIT",
            ]
        )

        if self._quadrature == True:
            ### use quadrature scheme
            self._lookup_filename_lw = lookup_filename_lw
            self._lookup_filename_sw = lookup_filename_sw

            # requires os.environ['ARTS_DATA_PATH'] to be set to the location of arts-cat-data
            line_basename = "lines/"
            xsec_basename = "xsec/"
            cia_basename = "cia-xml/hitran2011/hitran_cia2012_adapted.xml.gz"

            # No jacobian calculation
            self._ws[0].jacobianOff()

            # Clearsky = No scattering
            self._ws[0].cloudboxOff()

            # Output radiance not converted
            self._ws[0].StringSet(self._ws[0].iy_unit, "1")

            # set particle scattering to zero, because we want only clear sky
            self._ws[0].scat_data_checked = 1
            self._ws[0].Touch(self._ws[0].scat_data)

            # open quadrature/weight combo for the lw
            self._optimization_result_lw = xr.open_dataset(
                quadrature_filename_lw, engine="netcdf4"
            )

            self._S_lw = self._optimization_result_lw.S.data
            self._W_lw = self._optimization_result_lw.W.data

            self._ws[0].f_grid = pyarts.arts.convert.kaycm2freq(self._S_lw)

            # open quadrature/weight combo for the sw
            self._optimization_result_sw = xr.open_dataset(
                quadrature_filename_sw, engine="netcdf4"
            )

            self._S_sw = self._optimization_result_sw.S.data
            self._W_sw = self._optimization_result_sw.W.data

            self._ws[1].f_grid = pyarts.arts.convert.kaycm2freq(self._S_sw)

            if lookup_filename_lw == None:
                # Load MTCKD
                self._ws[0].ReadXML(
                    self._ws[0].predefined_model_data, "model/mt_ckd_4.0/H2O.xml"
                )

                # Read line catalog
                self._ws[0].abs_lines_per_speciesReadSpeciesSplitCatalog(
                    basename=line_basename
                )

                # Read cross section data
                self._ws[0].ReadXsecData(basename=xsec_basename)

                # Read CIA data
                self._ws[0].abs_cia_dataReadFromXML(filename=cia_basename)

                # Set line shape and cut off.
                self._ws[0].abs_lines_per_speciesCompact()  # Throw away lines outside f_grid
                self._ws[0].abs_lines_per_speciesLineShapeType(
                    self._ws[0].abs_lines_per_species, "VP"
                )
                self._ws[0].abs_lines_per_speciesNormalization(
                    self._ws[0].abs_lines_per_species, "VVH"
                )
                self._ws[0].abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
                self._ws[0].abs_lines_per_speciesTurnOffLineMixing()
                self._ws[0].propmat_clearsky_agendaAuto(T_extrapolfac=1e99, extpolfac=1e99)
            else:
                # Use lookup table
                abs_lookup_lw = lookup_filename_lw
                self._ws[0].abs_lookup_is_adapted.initialize_if_not()

                self._ws[0].abs_lines_per_speciesSetEmpty()
                self._ws[0].ReadXML(self._ws[0].abs_lookup, abs_lookup_lw)
                self._ws[0].f_gridFromGasAbsLookup()
                self._ws[0].abs_lookupAdapt()
                self._ws[0].propmat_clearsky_agendaAuto(
                    use_abs_lookup=1, T_extrapolfac=1e99, extpolfac=1e99
                )
            if lookup_filename_sw == None:
                # Load MTCKD
                self._ws[1].ReadXML(
                    self._ws[1].predefined_model_data, "model/mt_ckd_4.0/H2O.xml"
                )
                # Read line catalog
                self._ws[1].abs_lines_per_speciesReadSpeciesSplitCatalog(
                    basename=line_basename
                )

                # Read cross section data
                self._ws[1].ReadXsecData(basename=xsec_basename)

                # Read CIA data
                self._ws[1].abs_cia_dataReadFromXML(filename=cia_basename)

                self._ws[1].abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
                self._ws[1].abs_lines_per_speciesTurnOffLineMixing()
                self._ws[1].propmat_clearsky_agendaAuto(T_extrapolfac=1e99, extpolfac=1e99)
            else:
                # Load lookup table
                abs_lookup_sw = lookup_filename_sw
                self._ws[1].abs_lookup_is_adapted.initialize_if_not()
                self._ws[1].abs_lines_per_speciesSetEmpty()
                self._ws[1].ReadXML(self._ws[1].abs_lookup, abs_lookup_sw)
                self._ws[1].f_gridFromGasAbsLookup()
                self._ws[1].abs_lookupAdapt()

                self._ws[1].propmat_clearsky_agendaAuto(
                    use_abs_lookup=1, T_extrapolfac=1e99, extpolfac=1e99
                )
                self._ws[1].lbl_checked = 1
        else:
            ### use ARTS LBL with lookup table
            # Read lookup table
            abs_lookup_lw = os.getenv(
                "KONRAD_LOOKUP_TABLE_LW",
                join(dirname(__file__), "data/abs_lookup_lw.xml"),
            )

            abs_lookup_sw = os.getenv(
                "KONRAD_LOOKUP_TABLE_SW",
                join(dirname(__file__), "data/abs_lookup_sw.xml"),
            )

            ### lw
            self._ws[0].abs_lookup_is_adapted.initialize_if_not()
            if isfile(abs_lookup_lw):
                self._ws[0].abs_lines_per_speciesSetEmpty()
                self._ws[0].ReadXML(self._ws[0].abs_lookup, abs_lookup_lw)
                self._ws[0].f_gridFromGasAbsLookup()
                self._ws[0].abs_lookupAdapt()
                self._ws[0].propmat_clearsky_agendaAuto(
                    use_abs_lookup=1, T_extrapolfac=1e99, extpolfac=1e99
                )

                self._ws[0].jacobianOff()  # No jacobian calculation
                self._ws[0].cloudboxOff()  # Clearsky = No scattering
                self._ws[0].sensorOff()  # No sensor properties
            else:
                warnings.warn(
                    "Could not find ARTS absorption lookup table.\n"
                    "To perform ARTS calculations you have to download the lookup "
                    "table at:\n\n    https://doi.org/10.5281/zenodo.3885410\n\n"
                    "Afterwards, use the following environment variable to tell "
                    "konrad where to find it:\n\n"
                    "    $ export KONRAD_LOOKUP_TABLE_LW ='/path/to/abs_lookup_lw.xml'",
                    UserWarning,
                )
            ### sw
            self._ws[1].abs_lookup_is_adapted.initialize_if_not()
            if isfile(abs_lookup_sw):
                self._ws[1].abs_lines_per_speciesSetEmpty()
                self._ws[1].ReadXML(self._ws[1].abs_lookup, abs_lookup_sw)
                self._ws[1].f_gridFromGasAbsLookup()
                self._ws[1].abs_lookupAdapt()

                self._ws[1].propmat_clearsky_agendaAuto(
                    use_abs_lookup=1, T_extrapolfac=1e99, extpolfac=1e99
                )
                self._ws[1].lbl_checked = 1
            else:
                warnings.warn(
                    "Could not find ARTS absorption lookup table for the sw.\n"
                    "To perform ARTS calculations you need a lookup table."
                    "Use the following environment variable to tell "
                    "konrad where to find it:\n\n"
                    "    $ export KONRAD_LOOKUP_TABLE_SW='/path/to/abs_lookup_sw.xml'",
                    UserWarning,
                )
        # Set number of OMP threads
        if threads is not None:
            self._ws[0].SetNumberOfThreads(threads)
            self._ws[1].SetNumberOfThreads(threads)
            
    def calc_lookup_table_sw(
        self, 
        filename=None, 
        t_min=150., 
        t_max=350., 
        p_step=0.5,
        wavenumber=None):
        
        """Calculate an absorption lookup table for the shortwave.
        The table will cover surface temperatures between 150 and 350K, by default.
        The frequency grid covers the shortwave spectrom from 2000 to 50,000 cm^-1.
        
        Parameters:
            filename (str): where to store the lookup table
            t_min (float): minimum surface temperature to be covered
            t_max (float): maximum surface temperature to be covered
            p_step (float): how to discretize pressure
            wavenumber (array) [cm^-1]: array of wavenumbers.

        """
        if wavenumber is None:
            wavenumber = np.linspace(2000, 50000, 2**15)
            
        self._ws[1].f_grid = pyarts.arts.convert.kaycm2freq(wavenumber)
        
        # Read line catagloge and create absorption lines.
        self._ws[1].abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
        self._ws[1].ReadXML(self._ws[1].predefined_model_data, "model/mt_ckd_4.0/H2O.xml")
        self._ws[1].ReadXsecData(basename="xsec/")
        self._ws[1].abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
        
        # Setup lookup table calculation
        self._ws[1].abs_lookupSetupWide(t_min=t_min, t_max=t_max, p_step=p_step)
        
        # Setup propagation matrix agenda (absorption)
        self._ws[1].propmat_clearsky_agendaAuto(use_abs_lookup=0)
        self._ws[1].lbl_checked = 1
        
        self._ws[1].abs_lookupCalc()

        # save Lut
        if filename is not None:
            self._ws[1].WriteXML('binary', self._ws[1].abs_lookup, filename)
        

    def calc_lookup_table_lw(self, filename=None, fnum=2**15, wavenumber=None):
        """Calculate an absorption lookup table in the longwave.

        The lookup table is constructed to cover surface temperatures
        between 200 and 400 K, and water vapor mixing ratio up to 40%.

        The frequency grid covers the whole outgoing longwave spectrum
        from 10 to 3,250 cm^-1.

        References:
            An absorption lookup table can be found at
                https://doi.org/10.5281/zenodo.3885410

        Parameters:
            filename (str): (Optional) path to an ARTS XML file
                to store the lookup table.
            fnum (int): Number of frequencies in frequency grid.
                Ignored if `wavenumber` is set.
            wavenumber (ndarray): Wavenumber grid [m-1].
        """
        # Create a frequency grid
        if wavenumber is None:
            wavenumber = np.linspace(10e2, 3_250e2, fnum)
        self._ws[0].f_grid = ty.physics.wavenumber2frequency(wavenumber)

        # Read line catagloge and create absorption lines.
        self._ws[0].abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
        self._ws[0].ReadXML(self._ws[0].predefined_model_data, "model/mt_ckd_4.0/H2O.xml")

        # Set line shape and cut off.
        self._ws[0].abs_lines_per_speciesCompact()  # Throw away lines outside f_grid
        self._ws[0].abs_lines_per_speciesLineShapeType(self._ws[0].abs_lines_per_species, "VP")
        self._ws[0].abs_lines_per_speciesNormalization(self._ws[0].abs_lines_per_species, "VVH")
        self._ws[0].abs_lines_per_speciesCutoff(
            self._ws[0].abs_lines_per_species, "ByLine", 750e9
        )
        self._ws[0].abs_lines_per_speciesTurnOffLineMixing()
        self._ws[0].propmat_clearsky_agendaAuto(use_abs_lookup=0)

        # Create a standard atmosphere
        p_grid = get_quadratic_pgrid(1_200e2, 0.5, 80)

        atmosphere = Atmosphere(p_grid)
        atmosphere["T"][-1, :] = 300.0 + 5.0 * np.log(atmosphere["plev"] / 1000e2)
        atmosphere.tracegases_rcemip()
        atmosphere["O2"][:] = 0.2095
        atmosphere["CO2"][:] = 1.5 * 348e-6

        h2o = 0.01 * (p_grid / 1000e2) ** 0.2
        atmosphere["H2O"][:] = h2o[:-1]

        # Convert the konrad atmosphere into an ARTS atm_fields_compact.
        atm_fields_compact = atmosphere.to_atm_fields_compact()
        self._ws[0].atm_fields_compact = atm_fields_compact

        self._ws[0].atm_fields_compactAddConstant(
            atm_fields_compact=self._ws[0].atm_fields_compact,
            name="abs_species-N2",
            value=0.7808,
            condensibles=["abs_species-H2O"],
        )

        # Setup the lookup table calculation
        self._ws[0].AtmFieldsAndParticleBulkPropFieldFromCompact()
        self._ws[0].vmr_field.value = self._ws[0].vmr_field.value.value.clip(min=0.0)
        self._ws[0].atmfields_checkedCalc()
        self._ws[0].abs_lookupSetup(p_step=1.0)  # Do not refine p_grid
        self._ws[0].abs_t_pert = np.arange(-160, 61, 20)

        nls_idx = [
            i for i, tag in enumerate(self._ws[0].abs_species.value) if "H2O" in tag
        ][0]
        self._ws[0].abs_nls = [self._ws[0].abs_species.value[nls_idx]]

        self._ws[0].abs_nls_pert = np.array(
            [10**x for x in [-9, -7, -5, -3, -1, 0, 0.5, 1, 1.5, 2]]
        )

        # Run checks
        self._ws[0].propmat_clearsky_agenda_checkedCalc()
        self._ws[0].lbl_checkedCalc()

        # Calculate actual lookup table.
        self._ws[0].abs_lookupCalc()

        if filename is not None:
            self._ws[0].WriteXML("binary", self._ws[0].abs_lookup, filename)

    def set_atmospheric_state(self, atmosphere, t_surface):
        """Set and check the atmospheric fields."""
        import pyarts

        atm_fields_compact = atmosphere.to_atm_fields_compact()

        # sw geo
        self._ws[1].lat_true = [self._zenith_angle]
        self._ws[1].surface_scalar_reflectivity = [self._albedo]

        # Scale dry-air VMRs with H2O and CO2 content.
        if self.scale_vmr:
            variable_vmrs = (
                atm_fields_compact.get("abs_species-H2O")[0]
                + atm_fields_compact.get("abs_species-CO2")[0]
            )
        else:
            t3_shape = atm_fields_compact.get("abs_species-H2O")[0].shape
            variable_vmrs = np.zeros(t3_shape)
        for species in map(str, atm_fields_compact.grids[0]):
            if (
                species.startswith("abs_species-")
                and "H2O" not in species
                and "CO2" not in species
            ):
                atm_fields_compact.scale(species, 1 - variable_vmrs)
                
        # Compute the N2 VMR as a residual of the full atmosphere composition.
        n2 = pyarts.arts.GriddedField3(
            grids=atm_fields_compact.grids[1:],
            data=0.7808 * (1 - variable_vmrs),
        )

        self._ws[0].atm_fields_compact = atm_fields_compact
        self._ws[1].atm_fields_compact = atm_fields_compact

        self._ws[0].atm_fields_compactAddSpecies(
            atm_fields_compact=self._ws[0].atm_fields_compact,
            name="abs_species-N2",
            value=n2,
        )
        self._ws[1].atm_fields_compactAddSpecies(
            atm_fields_compact=self._ws[1].atm_fields_compact,
            name="abs_species-N2",
            value=n2,
        )

        self._ws[0].AtmFieldsAndParticleBulkPropFieldFromCompact()
        self._ws[0].vmr_field = self._ws[0].vmr_field.value.value.clip(min=0)

        self._ws[1].AtmFieldsAndParticleBulkPropFieldFromCompact()
        self._ws[1].vmr_field = self._ws[1].vmr_field.value.value.clip(min=0)

        # Surface & TOA
        # Add pressure layers to the surface and top-of-the-atmosphere to
        # ensure consistent atmosphere boundaries between ARTS and RRTMG.
        self._ws[0].t_surface = np.array([[t_surface]])
        self._ws[0].z_surface = np.array([[0.0]])
        self._ws[0].z_field.value[0, 0, 0] = 0.0
        self._ws[0].surface_skin_t = self._ws[0].t_field.value[0, 0, 0]

        self._ws[1].t_surface = np.array([[t_surface]])
        self._ws[1].z_surface = np.array([[0.0]])
        self._ws[1].z_field.value[0, 0, 0] = 0.0
        self._ws[1].surface_skin_t = self._ws[1].t_field.value[0, 0, 0]

        # sun
        self._ws[1].sunsOff()
        if -90 < self._zenith_angle < 90:
            self._ws[1].sunsAddSingleFromGrid(
                sun_spectrum_raw=self._sun_spectrum,
                temperature=0.0,
                latitude=0.0,
                longitude=0.0,
            )
        # set cloudbox to full atmosphere
        self._ws[0].cloudboxSetFullAtm()
        self._ws[0].pnd_fieldZero()

        # Perform configuration and atmosphere checks
        self._ws[0].atmfields_checkedCalc()
        self._ws[0].propmat_clearsky_agenda_checkedCalc()
        self._ws[0].atmgeom_checkedCalc()
        self._ws[0].cloudbox_checkedCalc()

        # sw
        # set cloudbox to full atmosphere
        self._ws[1].cloudboxSetFullAtm()

        # No jacobian calculations
        self._ws[1].jacobianOff()

        # set particle scattering

        self._ws[1].scat_data_checked = 1
        self._ws[1].Touch(self._ws[1].scat_data)
        self._ws[1].pnd_fieldZero()

    def calc_monochromatic_fluxes(self, atmosphere, t_surface):
        """Calculate the spectral irradiance field."""

        self.set_atmospheric_state(atmosphere, t_surface)

        if self._lookup_filename_lw == None:
            self._ws[0].scat_data_checkedCalc()
            self._ws[0].lbl_checkedCalc()
        # sw checks
        # Propagation path agendas and variables
        self._ws[1].NumericSet(self._ws[1].ppath_lmax, -1)
        self._ws[1].NumericSet(self._ws[1].ppath_lraytrace, 1e4)

        # Switch on/off gas scattering
        self._ws[1].IndexSet(self._ws[1].gas_scattering_do, 1)
        self._ws[1].IndexSet(self._ws[1].gas_scattering_output_type, 0)

        # Check model atmosphere
        self._ws[1].scat_data_checkedCalc()
        self._ws[1].atmfields_checkedCalc()
        self._ws[1].atmgeom_checkedCalc()
        self._ws[1].cloudbox_checkedCalc()
        self._ws[1].lbl_checkedCalc()

        ### perform longwave radiation calculation
        # get the zenith angle grid and the integrations weights
        self._ws[0].AngularGridsSetFluxCalc(
            N_za_grid=self.nstreams, N_aa_grid=1, za_grid_type="double_gauss"
        )

        # calculate intensity field
        self._ws[0].Tensor3Create("trans_field")

        self._ws[0].spectral_radiance_fieldClearskyPlaneParallel(
            trans_field=self._ws[0].trans_field,
            use_parallel_za=0,
        )
        self._ws[0].spectral_irradiance_fieldFromSpectralRadianceField()

        ### perform shortwave radiation calculation
        self._ws[1].DisortCalcIrradiance(emission=0)

        spec_flux_up_lw = self._ws[0].spectral_irradiance_field.value[:, :, 0, 0, 1]
        spec_flux_down_lw = self._ws[0].spectral_irradiance_field.value[:, :, 0, 0, 0]
        spec_flux_up_lw = spec_flux_up_lw * (c * 100)
        spec_flux_down_lw = spec_flux_down_lw * (c * 100)

        spec_flux_up_sw = self._ws[1].spectral_irradiance_field.value[:, :, 0, 0, 1]
        spec_flux_down_sw = self._ws[1].spectral_irradiance_field.value[:, :, 0, 0, 0]
        spec_flux_up_sw = spec_flux_up_sw * (c * 100)
        spec_flux_down_sw = spec_flux_down_sw * (c * 100)

        return (
            spec_flux_up_lw,
            -spec_flux_down_lw,
            spec_flux_up_sw,
            -spec_flux_down_sw,
        )

    def calc_spectral_irradiance_field(self, atmosphere, t_surface):
        """Calculate the spectral irradiance field."""
        if (
            not self._ws[0].abs_lookup_is_adapted.value.value
            and self._ws[1].abs_lookup_is_adapted.value.value
        ):
            raise Exception("Simulations without lookup table are not supported.")
        self.set_atmospheric_state(atmosphere, t_surface)

        # sw checks
        # Propagation path agendas and variables
        self._ws[1].NumericSet(self._ws[1].ppath_lmax, -1)
        self._ws[1].NumericSet(self._ws[1].ppath_lraytrace, 1e4)

        # Switch on/off gas scattering
        self._ws[1].IndexSet(self._ws[1].gas_scattering_do, 1)
        self._ws[1].IndexSet(self._ws[1].gas_scattering_output_type, 0)

        # Check model atmosphere
        self._ws[1].scat_data_checkedCalc()
        self._ws[1].atmfields_checkedCalc()
        self._ws[1].atmgeom_checkedCalc()
        self._ws[1].cloudbox_checkedCalc()
        self._ws[1].lbl_checkedCalc()

        ## perform sw radiation calculation
        self._ws[1].DisortCalcIrradiance(emission=0)

        ### perform lw radiation calculation
        # get the zenith angle grid and the integrations weights
        self._ws[0].AngularGridsSetFluxCalc(
            N_za_grid=self.nstreams, N_aa_grid=1, za_grid_type="double_gauss"
        )

        # calculate intensity field
        self._ws[0].Tensor3Create("trans_field")

        self._ws[0].spectral_radiance_fieldClearskyPlaneParallel(
            trans_field=self._ws[0].trans_field,
            use_parallel_za=0,
        )
        self._ws[0].spectral_irradiance_fieldFromSpectralRadianceField()

        return (
            self._ws[0].f_grid.value[:].copy(),
            self._ws[0].p_grid.value[:].copy(),
            self._ws[0].spectral_irradiance_field.value[:].copy(),
            self._ws[1].f_grid.value[:].copy(),
            self._ws[1].p_grid.value[:].copy(),
            self._ws[1].spectral_irradiance_field.value[:].copy(),
        )

    def calc_optical_thickness(self, atmosphere, t_surface):
        """Calculate the spectral irradiance field  in the lw."""
        self.set_atmospheric_state(atmosphere, t_surface)

        self._ws[0].propmat_clearsky_fieldCalc()

        tau = np.trapz(
            y=self._ws[0].propmat_clearsky_field.value[:, :, 0, 0, :, 0, 0],
            x=self._ws[0].z_field.value[:, 0, 0],
            axis=-1,
        )

        return self._ws[0].f_grid.value[:].copy(), tau

    def integrate_quadrature_irradiance(self, W, spectral_flux):
        int_flux = np.matmul(W, spectral_flux)
        return int_flux

    @staticmethod
    def integrate_spectral_irradiance(frequency, irradiance):
        """Integrate the spectral irradiance field over the frequency.

        Parameters:
            frequency (ndarray): Frequency [Hz].
            irradiance (ndarray): Spectral irradiance [W m^-2 / Hz].

        Returns:
            ndarray, ndarray: Downward flux, upward, flux [W m^-2]
        """
        F = np.trapz(irradiance, frequency, axis=0)[:, 0, 0, :]

        # Fluxes
        lw_down = -F[:, 0]
        lw_up = F[:, 1]

        return lw_down, lw_up

    def calc_spectral_olr(self, atmosphere, surface):
        """Calculate the outgoing longwave radiation as function of wavenumber.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
            surface (konrad.surface.Surface): Surface model.

        Returns:
           ndarray: Outgoing longwave radiation [W m^-2 / cm^-1]
        """
        f, _, irradiance_field, _ = self.calc_spectral_irradiance_field(
            atmosphere=atmosphere, t_surface=surface["temperature"][0]
        )
        return f, irradiance_field[:, -1, 0, 0, 1]


class ARTS(Radiation):
    def __init__(self, store_spectral_olr=False, *args, arts_kwargs={}, **kwargs):
        """Radiation class to provide line-by-line longwave fluxes.

        Parameters:
            store_spectral_olr (bool): Store spectral OLR in netCDF file.
                This will significantly increase the output size.
            args: Positional arguments are used to initialize
                `konrad.radiation.RRTMG`.
            arts_kwargs (dict): Keyword arguments that are used to initialize
                `konrad.radiation.arts._ARTS`.
            kwargs: Keyword arguments are used to initialize
                `konrad.radiation.RRTMG`.
        """
        super().__init__(*args, **kwargs)

        self.store_spectral_olr = store_spectral_olr
        self._arts = _ARTS(**arts_kwargs)

    def calc_radiation(self, atmosphere, surface, cloud):
        # Perform RRTMG simulation
        # Add a virtual layer ontop of the atmosphere column to improve the
        # accuracy of top-of-the-atmosphere fluxes.
        # The fluxes/heating rates in this level are ignored afterwards.
        self._arts._zenith_angle = self.current_solar_angle
        self._arts._albedo = surface.albedo

        if self._arts._quadrature == True:
            # Perform quadrature simulation
            (
                spec_flux_up_lw,
                spec_flux_down_lw,
                spec_flux_up_sw,
                spec_flux_down_sw,
            ) = self._arts.calc_monochromatic_fluxes(
                atmosphere=atmosphere, t_surface=surface["temperature"][0]
            )

            Fd_lw = self._arts.integrate_quadrature_irradiance(
                W=self._arts._W_lw, spectral_flux=spec_flux_down_lw
            )
            Fu_lw = self._arts.integrate_quadrature_irradiance(
                W=self._arts._W_lw, spectral_flux=spec_flux_up_lw
            )
            Fd_sw = self._arts.integrate_quadrature_irradiance(
                W=self._arts._W_sw, spectral_flux=spec_flux_down_sw
            )
            Fu_sw = self._arts.integrate_quadrature_irradiance(
                W=self._arts._W_sw, spectral_flux=spec_flux_up_sw
            )
        else:
            # Perform ARTS simulation
            (
                f_lw,
                _,
                irradiance_field_lw,
                f_sw,
                _,
                irradiance_field_sw,
            ) = self._arts.calc_spectral_irradiance_field(
                atmosphere=atmosphere, t_surface=surface["temperature"][0]
            )
            Fd_lw, Fu_lw = self._arts.integrate_spectral_irradiance(
                f_lw, irradiance_field_lw
            )
            Fd_sw, Fu_sw = self._arts.integrate_spectral_irradiance(
                f_sw, irradiance_field_sw
            )

        # Interpolate RT results on fine original grid
        def _reshape(x, trim=-1):
            return x[:trim].reshape(1, -1)

        self["lw_flxu"] = _reshape(Fu_lw, trim=None)
        self["lw_flxd"] = _reshape(Fd_lw, trim=None)
        self["lw_flxu_clr"] = _reshape(Fu_lw, trim=None)
        self["lw_flxd_clr"] = _reshape(Fd_lw, trim=None)
        self["sw_flxu"] = _reshape(Fu_sw, trim=None)
        self["sw_flxd"] = _reshape(Fd_sw, trim=None)
        self["sw_flxu_clr"] = _reshape(Fu_sw, trim=None)
        self["sw_flxd_clr"] = _reshape(Fd_sw, trim=None)

        self["lw_htngrt"] = np.zeros((1, atmosphere["plev"].size))
        self["lw_htngrt_clr"] = np.zeros((1, atmosphere["plev"].size))
        self["sw_htngrt"] = np.zeros((1, atmosphere["plev"].size))
        self["sw_htngrt_clr"] = np.zeros((1, atmosphere["plev"].size))

        self.coords = {
            "time": np.array([0]),
            "phlev": atmosphere["phlev"],
            "plev": atmosphere["plev"],
        }

        if self.store_spectral_olr:
            self.coords.update({"frequency": f})
            self.create_variable(
                name="outgoing_longwave_radiation",
                data=irradiance_field[:, -1, 0, 0, 1].reshape(1, -1),
                dims=("time", "frequency"),
            )

    def update_heatingrates(self, atmosphere, surface, cloud=None):
        """Returns `xr.Dataset` containing radiative transfer results."""
        self.calc_radiation(atmosphere, surface, cloud)

        def fluxes(net_fluxes, pressure):
            Q = fluxes2heating(net_fluxes, pressure, method="gradient")
            f = PchipInterpolator(np.log(pressure[::-1]), Q[::-1])
            return f(np.log(atmosphere["plev"]))

        self["sw_htngrt"][-1] = fluxes(
            net_fluxes=self["sw_flxu"][-1] - self["sw_flxd"][-1],
            pressure=atmosphere["phlev"],
        )

        self["sw_htngrt_clr"][-1] = fluxes(
            net_fluxes=self["sw_flxu_clr"][-1] - self["sw_flxd_clr"][-1],
            pressure=atmosphere["phlev"],
        )

        self["lw_htngrt"][-1] = fluxes(
            net_fluxes=self["lw_flxu"][-1] - self["lw_flxd"][-1],
            pressure=atmosphere["phlev"],
        )

        self["lw_htngrt_clr"][-1] = fluxes(
            net_fluxes=self["lw_flxu_clr"][-1] - self["lw_flxd_clr"][-1],
            pressure=atmosphere["phlev"],
        )

        self.derive_diagnostics()
