"""Define an interface for the PSRAD radiation scheme. """
import numpy as np
import xarray as xr

from .radiation import Radiation


__all__ = [
    'PSRAD',
]


class PSRAD(Radiation):
    """Radiation model using the ICON PSRAD radiation scheme.

    The PSRAD class relies on external radiative-transfer code. Currently this
    functionality is provided by the PSRAD radiation scheme. You need a
    compiled version of this code in order to install and run ``konrad``.

    A stable version is accessible through the internal subversion repository:

        $ svn co https://arts.mi.uni-hamburg.de/svn/internal/psrad/trunk psrad

    Follow the instructions given in the repository to compile PSRAD on your
    machine. A part of the installation process is to set some environment
    variables. Thos are also needed in order to run ``konrad``:

        $ source config/psrad_env.bash

    """
    @staticmethod
    def _extract_psrad_args(atmosphere):
        """Returns tuple of mixing ratios to use with psrad.

        Paramteres:
            atmosphere (dict or pandas.DataFrame): Atmospheric atmosphere.

        Returns:
            tuple(ndarray): ndarrays in the order and unit to use with `psrad`:
                Z, P, T, x_vmr, ...
        """
        z = atmosphere['z'].values
        p = atmosphere['plev'].values / 100
        T = atmosphere['T'].values

        ret = [z, p, T]  # Z, P, T

        # Keep order as it is expected by PSRAD.
        required_gases = ['H2O', 'O3', 'CO2', 'N2O', 'CO', 'CH4']

        for gas in required_gases:
            if gas in atmosphere:
                ret.append(atmosphere[gas].values * 1e6)  # Append gas in ppm.
            else:
                ret.append(np.zeros(np.size(p)))

        return tuple(ret)

    def calc_radiation(self, atmosphere):
        """Returns the shortwave, longwave and net heatingrates.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): Atmosphere model.

        Returns:
            xarray.Dataset: Dataset containing for the simulated heating rates.
                The keys are 'sw_htngrt', 'lw_htngrt' and 'net_htngrt'.
        """
        from . import _psrad
        self.psrad = _psrad

        dmy_indices = np.asarray([0, 0, 0, 0])
        ic = dmy_indices.astype("int32") + 1
        c_lwc = np.asarray([0., 0., 0., 0.])
        c_iwc = np.asarray([0., 0., 0., 0.])
        c_frc = np.asarray([0., 0., 0., 0.])

        nlev = atmosphere['plev'].size

        # Extract surface properties.
        P_sfc = atmosphere.surface.pressure.values / 100
        T_sfc = atmosphere.surface.temperature.values[0]
        albedo = atmosphere.surface.albedo.values

        # Use the **current** solar angle as zenith angle for the simulation.
        zenith = self.current_solar_angle

        self.psrad.setup_single(nlev, len(ic), ic, c_lwc, c_iwc, c_frc, zenith,
                                albedo, P_sfc, T_sfc,
                                *self._extract_psrad_args(atmosphere))

        self.psrad.advance_lrtm()  # Longwave simulations.
        (lw_hr, lw_hr_clr, lw_flxd,
         lw_flxd_clr, lw_flxu, lw_flxu_clr) = self.psrad.get_lw_fluxes()

        self.psrad.advance_srtm()  # Shortwave simulations.

        (sw_hr, sw_hr_clr, sw_flxd,
         sw_flxd_clr, sw_flxu, sw_flxu_clr,
         vis_frc, par_dn, nir_dff, vis_diff,
         par_diff) = self.psrad.get_sw_fluxes()

        ret = xr.Dataset({
            # General atmospheric properties.
            'z': atmosphere['z'],
            # Longwave fluxes and heatingrates.
            'lw_htngrt': (['time', 'plev'], lw_hr[:, :]),
            'lw_htngrt_clr': (['time', 'plev'], lw_hr_clr[:, :]),
            'lw_flxu': (['time', 'phlev'], lw_flxu[:, :]),
            'lw_flxd': (['time', 'phlev'], lw_flxd[:, :]),
            'lw_flxu_clr': (['time', 'phlev'], lw_flxu_clr[:, :]),
            'lw_flxd_clr': (['time', 'phlev'], lw_flxd_clr[:, :]),
            # Shortwave fluxes and heatingrates.
            # Note: The shortwave fluxes and heatingrates calculated by PSRAD
            # are **inverted**. Therefore, they are flipped to make the input
            # and output of this function consistent.
            'sw_htngrt': (['time', 'plev'], sw_hr[:, ::-1]),
            'sw_htngrt_clr': (['time', 'plev'], sw_hr_clr[:, ::-1]),
            'sw_flxu': (['time', 'phlev'], sw_flxu[:, ::-1]),
            'sw_flxd': (['time', 'phlev'], sw_flxd[:, ::-1]),
            'sw_flxu_clr': (['time', 'phlev'], sw_flxu_clr[:, ::-1]),
            'sw_flxd_clr': (['time', 'phlev'], sw_flxd_clr[:, ::-1]),
        },
            coords={
                'time': [0],
                'plev': atmosphere['plev'].values,
                'phlev': atmosphere['phlev'].values,
            }
        )

        return ret
