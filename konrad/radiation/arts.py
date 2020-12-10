import logging
import os
from os.path import join, dirname, isfile

import numpy as np
import typhon as ty
from scipy.interpolate import PchipInterpolator

from konrad.utils import get_quadratic_pgrid
from konrad.atmosphere import Atmosphere
from konrad.cloud import ClearSky
from .rrtmg import RRTMG
from .common import fluxes2heating


logger = logging.getLogger(__name__)


class _ARTS:
    def __init__(self, ws=None, threads=None, nstreams=4, verbosity=0):
        """Initialize a wrapper for an ARTS workspace.

        Parameters:
            ws (pyarts.workspace.Workspace): An ARTS workspace.
            threads (int): Number of threads to use.
                Default is all available threads.
            nstreams (int): Number of viewing angles to base the radiative
                flux calculation on.
            verbosity (int): Control the ARTS verbosity from 0 (quiet) to 2.
        """
        from pyarts.workspace import Workspace, arts_agenda

        self.nstreams = nstreams

        if ws is None:
            self.ws = Workspace(verbosity=verbosity)

        self.ws.execute_controlfile("general/general.arts")
        self.ws.execute_controlfile("general/continua.arts")
        self.ws.execute_controlfile("general/agendas.arts")
        self.ws.execute_controlfile("general/planet_earth.arts")

        # Agenda settings
        self.ws.Copy(self.ws.abs_xsec_agenda, self.ws.abs_xsec_agenda__noCIA)
        self.ws.Copy(self.ws.iy_main_agenda, self.ws.iy_main_agenda__Emission)
        self.ws.Copy(self.ws.iy_space_agenda, self.ws.iy_space_agenda__CosmicBackground)
        self.ws.Copy(
            self.ws.iy_surface_agenda, self.ws.iy_surface_agenda__UseSurfaceRtprop
        )
        self.ws.Copy(
            self.ws.propmat_clearsky_agenda,
            self.ws.propmat_clearsky_agenda__LookUpTable,
        )
        self.ws.Copy(self.ws.ppath_agenda, self.ws.ppath_agenda__FollowSensorLosPath)
        self.ws.Copy(
            self.ws.ppath_step_agenda, self.ws.ppath_step_agenda__GeometricPath
        )

        @arts_agenda
        def p_eq_agenda(workspace):
            workspace.water_p_eq_fieldMK05()

        self.ws.Copy(self.ws.water_p_eq_agenda, p_eq_agenda)

        @arts_agenda
        def cloudbox_agenda(workspace):
            workspace.iyInterpCloudboxField()

        self.ws.Copy(self.ws.iy_cloudbox_agenda, cloudbox_agenda)

        # Number of Stokes components to be computed
        self.ws.IndexSet(self.ws.stokes_dim, 1)

        self.ws.jacobianOff()  # No jacobian calculation
        self.ws.cloudboxOff()  # Clearsky = No scattering

        # Set Absorption Species
        self.ws.abs_speciesSet(
            species=[
                "O2, O2-CIAfunCKDMT100",
                "H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
                "O3",
                "CO2, CO2-CKDMT252",
                "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
                "N2O",
                "CH4",
                "CO",
            ]
        )

        # Surface handling
        self.ws.VectorSetConstant(self.ws.surface_scalar_reflectivity, 1, 0.0)
        self.ws.Copy(
            self.ws.surface_rtprop_agenda,
            self.ws.surface_rtprop_agenda__Specular_NoPol_ReflFix_SurfTFromt_surface,
        )

        # Read lookup table
        abs_lookup = os.getenv(
            "KONRAD_LOOKUP_TABLE",
            join(dirname(__file__), "data/abs_lookup.xml")
        )

        if not isfile(abs_lookup):
            raise FileNotFoundError(
                "Could not find ARTS absorption lookup table.\n"
                "To perform ARTS calculations you have to download the lookup "
                "table at:\n\n    https://doi.org/10.5281/zenodo.3885410\n\n"
                "Afterwards, use the following environment variable to tell "
                "konrad where to find it:\n\n"
                "    $ export KONRAD_LOOKUP_TABLE='/path/to/abs_lookup.xml'"
            )

        self.ws.ReadXML(self.ws.abs_lookup, abs_lookup)
        self.ws.f_gridFromGasAbsLookup()
        self.ws.abs_lookupAdapt()

        # Sensor settings
        self.ws.sensorOff()  # No sensor properties

        # Atmosphere
        self.ws.AtmosphereSet1D()

        # Set number of OMP threads
        if threads is not None:
            self.ws.SetNumberOfThreads(threads)

    def calc_lookup_table(self, filename=None):
        """Calculate an absorption lookup table.

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
        """
        # Create a frequency grid
        wavenumber = np.linspace(10e2, 3_250e2, 2**15)  # 1 to 3000cm^-1
        self.ws.f_grid = ty.physics.wavenumber2frequency(wavenumber)

        # Read line catagloge and create absorption lines.
        self.ws.ReadSplitARTSCAT(
            abs_lines=self.ws.abs_lines,
            abs_species=self.ws.abs_species,
            basename="hitran_split_artscat5/",
            fmin=0.0,
            fmax=1e99,
            globalquantumnumbers="",
            localquantumnumbers="",
            ignore_missing=0,
        )

        # Set line shape and cut off.
        self.ws.abs_linesSetLineShapeType(self.ws.abs_lines, "VP")
        self.ws.abs_linesSetNormalization(self.ws.abs_lines, "VVH")
        self.ws.abs_linesSetCutoff(self.ws.abs_lines, "ByLine", 750e9)

        self.ws.abs_lines_per_speciesCreateFromLines()
        self.ws.abs_lines_per_speciesCompact()

        # Create a standard atmosphere
        p_grid = get_quadratic_pgrid(1_200e2, 0.5, 128)

        atmosphere = Atmosphere(p_grid)
        atmosphere["T"][:] = atmosphere["T"].clip(min=200)
        atmosphere.tracegases_rcemip()
        atmosphere["O2"][:] = 0.2095
        atmosphere["CO2"][:] = 1.5 * 348e-6

        h2o = 0.04 * (p_grid / p_grid[0])**0.2
        atmosphere["H2O"][:] = h2o[:-1]

        # Convert the konrad atmosphere into an ARTS atm_fields_compact.
        atm_fields_compact = atmosphere.to_atm_fields_compact()
        self.ws.atm_fields_compact = atm_fields_compact

        self.ws.atm_fields_compactAddConstant(
            atm_fields_compact=self.ws.atm_fields_compact,
            name="abs_species-N2",
            value=0.7808,
            condensibles=["abs_species-H2O"],
        )

        # Setup the lookup table calculation
        self.ws.AtmFieldsAndParticleBulkPropFieldFromCompact()
        self.ws.vmr_field.value = self.ws.vmr_field.value.clip(min=0.0)
        self.ws.atmfields_checkedCalc()
        self.ws.abs_lookupSetup(p_step=1.0)  # Do not refine p_grid
        self.ws.abs_t_pert = np.arange(-100, 101, 25)

        nls_idx = [i for i, tag in enumerate(self.ws.abs_species.value)
                   if "H2O" in tag[0]]
        self.ws.abs_speciesSet(
                abs_species=self.ws.abs_nls,
                species=[", ".join(self.ws.abs_species.value[nls_idx[0]])],
        )

        self.ws.abs_nls_pert = np.array([10**n for n in range(-4, 2)])

        # Run checks
        self.ws.abs_xsec_agenda_checkedCalc()
        self.ws.lbl_checkedCalc()

        # Calculate actual lookup table.
        self.ws.abs_lookupCalc()

        if filename is not None:
            self.ws.WriteXML("binary", self.ws.abs_lookup, filename)

    def calc_spectral_irradiance_field(self, atmosphere, t_surface):
        """Calculate the spectral irradiance field."""
        atm_fields_compact = atmosphere.to_atm_fields_compact()

        # Scale dry air VMRs with water content
        vmr_h2o = atm_fields_compact.get("abs_species-H2O")
        total_vmr = vmr_h2o[0]
        for species in atm_fields_compact.grids[0]:
            if species.startswith("abs_species-") and "H2O" not in species:
                atm_fields_compact.scale(species, 1 - vmr_h2o)
                total_vmr += atm_fields_compact.get(species)[0]

        # Compute the N2 VMR as a residual of the full atmosphere composition.
        n2 = ty.arts.types.GriddedField3(
            grids=atm_fields_compact.grids[1:],
            data=1 - total_vmr,
        )

        self.ws.atm_fields_compact = atm_fields_compact
        self.ws.atm_fields_compactAddSpecies(
            atm_fields_compact=self.ws.atm_fields_compact,
            name="abs_species-N2",
            value=n2,
        )
        self.ws.AtmFieldsAndParticleBulkPropFieldFromCompact()
        self.ws.vmr_field = self.ws.vmr_field.value.clip(min=0)

        # Surface & TOA
        # Add pressure layers to the surface and top-of-the-atmosphere to
        # ensure consistent atmosphere boundaries between ARTS and RRTMG.
        self.ws.t_surface = np.array([[t_surface]])
        self.ws.z_surface = np.array([[0.0]])
        self.ws.z_field.value[0, 0, 0] = 0.0

        # Perform RT calculations
        self.ws.atmfields_checkedCalc(bad_partition_functions_ok=1)
        self.ws.propmat_clearsky_agenda_checkedCalc()
        self.ws.atmgeom_checkedCalc()
        self.ws.cloudbox_checkedCalc()

        # get the zenith angle grid and the integrations weights
        self.ws.AngularGridsSetFluxCalc(
            N_za_grid=self.nstreams,
            N_aa_grid=1,
            za_grid_type="double_gauss"
        )

        # calculate intensity field
        self.ws.Tensor3Create("trans_field")
        self.ws.spectral_radiance_fieldClearskyPlaneParallel(
            trans_field=self.ws.trans_field, use_parallel_iy=1
        )
        self.ws.spectral_irradiance_fieldFromSpectralRadianceField()

        return (
            self.ws.f_grid.value.copy(),
            self.ws.p_grid.value.copy(),
            self.ws.spectral_irradiance_field.value.copy(),
            self.ws.trans_field.value[:, 1:, 0].copy().prod(axis=1),
        )

    def calc_radiative_fluxes(self, atmosphere, surface):
        """Calculate radiative fluxes.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.
            surface (konrad.surface.Surface): Surface model.

        Returns:
            ndarray, ndarray: Downward flux, upward, flux [W m^-2]
        """
        f, plev, irradiance_field, _ = self.calc_spectral_irradiance_field(
            atmosphere=atmosphere, t_surface=surface["temperature"][0]
        )
        F = np.trapz(irradiance_field, f, axis=0)[:, 0, 0, :]

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


class ARTS(RRTMG):
    def __init__(self, *args, arts_kwargs={}, **kwargs):
        """Radiation class to provide line-by-line longwave fluxes.

        Parameters:
            args: Positional arguments are used to initialize
                `konrad.radiation.RRTMG`.
            arts_kwargs (dict): Keyword arguments that are used to initialize
                `konrad.radiation.arts._ARTS`.
            kwargs: Keyword arguments are used to initialize
                `konrad.radiation.RRTMG`.
        """
        super().__init__(*args, **kwargs)

        self._arts = _ARTS(**arts_kwargs)

    def calc_radiation(self, atmosphere, surface, cloud):
        # Perform RRTMG simulation
        # Add a virtual layer ontop of the atmosphere column to improve the
        # accuracy of top-of-the-atmosphere fluxes.
        # The fluxes/heating rates in this level are ignored afterwards.
        ph_rrtmg = np.append(atmosphere["phlev"], 1e-2)
        atmosphere_rrtmg = atmosphere.refine_plev(ph_rrtmg, kind="nearest")

        lw_dT_fluxes, sw_dT_fluxes = self.radiative_fluxes(
            atmosphere_rrtmg,
            surface,
            ClearSky.from_atmosphere(atmosphere_rrtmg),
        )
        sw_fluxes = sw_dT_fluxes[1]

        # Perform ARTS simulation
        Fd, Fu = self._arts.calc_radiative_fluxes(atmosphere, surface)

        # Interpolate RT results on fine original grid
        def _reshape(x, trim=-1):
            return x[:trim].reshape(1, -1)

        self['lw_flxu'] = _reshape(Fu, trim=None)
        self['lw_flxd'] = _reshape(Fd, trim=None)
        self['lw_flxu_clr'] = _reshape(Fu, trim=None)
        self['lw_flxd_clr'] = _reshape(Fd, trim=None)
        self['sw_flxu'] = _reshape(
            sw_fluxes['upwelling_shortwave_flux_in_air'].data)
        self['sw_flxd'] = _reshape(
            sw_fluxes['downwelling_shortwave_flux_in_air'].data)
        self['sw_flxu_clr'] = _reshape(
            sw_fluxes['upwelling_shortwave_flux_in_air_assuming_clear_sky'].data)
        self['sw_flxd_clr'] = _reshape(
            sw_fluxes['downwelling_shortwave_flux_in_air_assuming_clear_sky'].data)

        self['lw_htngrt'] = np.zeros((1, atmosphere["plev"].size))
        self['lw_htngrt_clr'] = np.zeros((1, atmosphere["plev"].size))
        self['sw_htngrt'] = np.zeros((1, atmosphere["plev"].size))
        self['sw_htngrt_clr'] = np.zeros((1, atmosphere["plev"].size))

        self.coords = {
            'time': np.array([0]),
            'phlev': atmosphere['phlev'],
            'plev': atmosphere['plev'],
        }

    def update_heatingrates(self, atmosphere, surface, cloud):
        """Returns `xr.Dataset` containing radiative transfer results."""
        self.calc_radiation(atmosphere, surface, cloud)

        def fluxes(net_fluxes, pressure):
            Q = fluxes2heating(net_fluxes, pressure, method="gradient")
            f = PchipInterpolator(np.log(pressure[::-1]), Q[::-1])
            return f(np.log(atmosphere["plev"]))

        self['sw_htngrt'][-1] = fluxes(
            net_fluxes=self['sw_flxu'][-1] - self['sw_flxd'][-1],
            pressure=atmosphere['phlev'],
        )

        self['sw_htngrt_clr'][-1] = fluxes(
            net_fluxes=self['sw_flxu_clr'][-1] - self['sw_flxd_clr'][-1],
            pressure=atmosphere['phlev'],
        )

        self['lw_htngrt'][-1] = fluxes(
            net_fluxes=self['lw_flxu'][-1] - self['lw_flxd'][-1],
            pressure=atmosphere['phlev'],
        )

        self['lw_htngrt_clr'][-1] = fluxes(
            net_fluxes=self['lw_flxu_clr'][-1] - self['lw_flxd_clr'][-1],
            pressure=atmosphere['phlev'],
        )

        self.derive_diagnostics()
