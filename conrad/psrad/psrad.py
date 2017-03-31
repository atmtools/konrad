"""Module PSRAD: ICON radiation library interface.
The functions in this module provide interface to the PSRAD radiation library used in the ICON model. The simplest usage consists of:

* calling "setup" with all relevant input data;
* calling "advance_lrtm" and/or "advance_srtm" to compute the respective fluxes;
* calling "get_lw_fluxes" and/or "get_sw_fluxes" to obtain the results.

There are, in addition, query functions for variables internal to the module.

The package must be able to find "libpsrad.so.1" and its associated "*.nc" data files, all of which should reside in the same path from which python is first invoked.

This interface uses the Numpy module, and all arrays passed as arguments to or returned by functions are Numpy arrays. Integer and floating-point arrays must be of types "int32" and "float64" respectively, othewise the module will raise and exception.
"""
from ctypes import cdll as __cdll
from ctypes import POINTER as __POINTER
from ctypes import c_int, c_double
solib = __cdll.LoadLibrary('./libpsrad.so.1')
_c_int_p = __POINTER(c_int)
_c_double_p = __POINTER(c_double)

def __get_double_p(a):
  from numpy import dtype
  from ctypes import POINTER
  if (a.dtype != dtype('float64')):
    print("parameter is ", a.dtype.str, "(expected float64)")
    raise()
  return a.ctypes.data_as(POINTER(c_double))

def __get_int_p(a):
  from numpy import dtype
  from ctypes import POINTER
  if (a.dtype != dtype('int32')):
    print("parameter is ", a.dtype.str, "(expected int32)")
    raise()
  return a.ctypes.data_as(POINTER(c_int))

solib.psrad_get_kbdim.restype = c_int
def get_kbdim():
  "Returns (int32) the number of columns, as specified in the call to 'setup'."
  return solib.psrad_get_kbdim()

solib.psrad_get_klev.restype = c_int
def get_klev():
  "Returns (int32) the number of layers, as specified in the call to 'setup'."
  return solib.psrad_get_klev()

solib.psrad_get_nlayers.restype = c_int
def get_nlayers():
  "Returns (int32) the number of cloud layers, as specified in the call to 'setup'"
  return solib.psrad_get_nlayers()

def get_strategy_lw():
  "Returns (int32) the LW sampling strategy. Currently hard-coded."
  return 1

def get_strategy_sw():
  "Returns (int32) the SW sampling strategy. Currently hard-coded."
  return 1

solib.psrad_get_num_gpts_lw.restype = c_int
def get_num_gpts_lw():
  "Returns (int32) the total number of g-points in the long-wave range."
  return solib.psrad_get_ngptlw()

solib.psrad_get_num_gpts_sw.restype = c_int
def get_num_gpts_sw():
  "Returns (int32) the total number of g-points in the short-wave range."
  return solib.psrad_get_ngptsw()

solib.psrad_get_nbndlw.restype = c_int
def get_nbandlw():
  "Returns (int32) the number of long-wave bands. Currently hard-coded."
  return solib.psrad_get_nbndlw()

solib.psrad_get_nbndsw.restype = c_int
def get_nbandsw():
  "Returns (int32) the number of short-wave bands. Currently hard-coded."
  return solib.psrad_get_nbndsw()

solib.psrad_get_prmu0.restype = c_double
def get_prmu0():
  "Returns (float64) cos(zenith) for the value specified in the call to 'setup'."
  return solib.psrad_get_prmu0()


solib.psrad_standalone_setup_single.argtypes = [c_int, \
c_int, _c_int_p, _c_double_p, _c_double_p, _c_double_p, c_double, c_double, \
c_double, c_double, _c_double_p, _c_double_p, _c_double_p, _c_double_p, \
_c_double_p, _c_double_p, _c_double_p,_c_double_p]
def setup_single(klev, nlayers, icldlyr_index, cldlyr_lwc, cldlyr_iwc, \
cldlyr_frc, zenith, albedo, pp_sfc, tk_sfc, hgt_fl_vr, pp_fl_vr, tk_fl_vr, \
xm_h2o_vr, xm_o3_vr, xm_co2_vr, xm_n2o_vr, xm_co_vr, xm_ch4_vr):
  """Setup radiation calculation. Input data is as follows:
* klev (int): number of layers in the column
* nlayers (int): number of layers with clouds
* icldlyr_index (int[nlayers]): indexes (in [1:klev]) of layers with clouds. NOTICE THESE INDICES ARE IN FORTRAN 1-OFFSET CONVENTION.
* cldlyr_lwc (float64[nlayers]): Liquid water content in cloud layer. [???]
* cldlyr_iwc (float64[nlayers]): Ice water content in cloud layer. [???]
* cldlyr_frc (float64[nlayers]): Cloud fraction in cloud layer (in [0,1]).
* zenith (float64): Solar zenith angle [deg]
* albedo (float64): constant, band independent albedo (in [0,1]).
* pp_sfc (float64): Surface pressure [Pa]
* tk_sfc (float64): Surface temperature [K]
* hgt_fl_vr (float64[klev]): Full level heights [mb]
* pp_fl_vr (float64[klev]): Full level pressure [Pa]
* tk_fl_vr (float64[klev]): Full level temperature [K]
* xm_*_vr (float64[klev]): Volume mixing rations for chemical species *. [???]
The last variables are indexed from the ground up, i.e. pp_fl_vr[0] is the pressure at the surface and pp_fl_vr[klev-1] is the TOA pressure."""
  l_lw = c_int(1)
  l_sw = c_int(1)
  klev = c_int(klev)
  nlayers = c_int(len(icldlyr_index))

  C_icldlyr_index = __get_int_p(icldlyr_index)
  C_cldlyr_lwc = __get_double_p(cldlyr_lwc) 
  C_cldlyr_iwc = __get_double_p(cldlyr_iwc)
  C_cldlyr_frc = __get_double_p(cldlyr_frc)
  C_pp_sfc = c_double(pp_sfc)
  C_tk_sfc = c_double(tk_sfc)
  C_hgt_fl_vr = __get_double_p(hgt_fl_vr)
  C_pp_fl_vr = __get_double_p(pp_fl_vr)
  C_tk_fl_vr = __get_double_p(tk_fl_vr)
  C_xm_h2o_vr = __get_double_p(xm_h2o_vr)
  C_xm_o3_vr = __get_double_p(xm_o3_vr)
  C_xm_co2_vr = __get_double_p(xm_co2_vr)
  C_xm_n2o_vr = __get_double_p(xm_n2o_vr)
  C_xm_co_vr = __get_double_p(xm_co_vr)
  C_xm_ch4_vr = __get_double_p(xm_ch4_vr)
  C_zenith = c_double(zenith)
  C_albedo = c_double(albedo)
  
  solib.psrad_standalone_setup_single(klev, nlayers, C_icldlyr_index, C_cldlyr_lwc, C_cldlyr_iwc, C_cldlyr_frc, C_zenith, C_albedo, C_pp_sfc, C_tk_sfc, C_hgt_fl_vr, C_pp_fl_vr, C_tk_fl_vr, C_xm_h2o_vr, C_xm_o3_vr, C_xm_co2_vr, C_xm_n2o_vr, C_xm_co_vr, C_xm_ch4_vr)

solib.psrad_standalone_setup_multi.argtypes = [c_int, c_int, _c_int_p, \
_c_double_p, _c_double_p, _c_double_p, c_double, c_double, \
_c_double_p, _c_double_p, _c_double_p, _c_double_p, _c_double_p, _c_double_p, \
_c_double_p, _c_double_p, _c_double_p,_c_double_p]
def setup_multi(kbdim, klev, icldlyr, cldlyr_lwc, cldlyr_iwc, \
cldlyr_frc, zenith, albedo, pp_sfc, tk_sfc, hgt_fl_vr, pp_fl_vr, tk_fl_vr, \
xm_h2o_vr, xm_o3_vr, xm_co2_vr, xm_n2o_vr, xm_co_vr, xm_ch4_vr):
  """Setup radiation calculation. Input data is as follows:
* kbdim (int): number of columns
* klev (int): number of layers per column
* icldlyr (int[kbdim,klev]): flag for cells with clouds.
* cldlyr_lwc (float64[kbdim,klev]): Liquid water content in cloud layer. [???]
* cldlyr_iwc (float64[kbdim,klev]): Ice water content in cloud layer. [???]
* cldlyr_frc (float64[kbdim,klev]): Cloud fraction in cloud layer (in [0,1]).
* zenith (float64): Solar zenith angle [deg]
* albedo (float64): constant, band independent albedo (in [0,1]).
* pp_sfc (float64[kbdim]): Surface pressure [Pa]
* tk_sfc (float64[kbdim]): Surface temperature [K]
* hgt_fl_vr (float64[kbdim,klev]): Full level heights [mb]
* pp_fl_vr (float64[kbdim,klev]): Full level pressure [Pa]
* tk_fl_vr (float64[kbdim,klev]): Full level temperature [K]
* xm_*_vr (float64[kbdim,klev]): Volume mixing rations for chemical species *. [???]
The last variables are indexed from the ground up, i.e. pp_fl_vr[:,0] is the pressure at the surface and pp_fl_vr[:,klev-1] is the TOA pressure."""
  l_lw = c_int(1)
  l_sw = c_int(1)
  kbdim = c_int(kbdim)
  klev = c_int(klev)

  C_icldlyr = __get_int_p(icldlyr)
  C_cldlyr_lwc = __get_double_p(cldlyr_lwc) 
  C_cldlyr_iwc = __get_double_p(cldlyr_iwc)
  C_cldlyr_frc = __get_double_p(cldlyr_frc)
  C_pp_sfc = __get_double_p(pp_sfc)
  C_tk_sfc = __get_double_p(tk_sfc)
  C_hgt_fl_vr = __get_double_p(hgt_fl_vr)
  C_pp_fl_vr = __get_double_p(pp_fl_vr)
  C_tk_fl_vr = __get_double_p(tk_fl_vr)
  C_xm_h2o_vr = __get_double_p(xm_h2o_vr)
  C_xm_o3_vr = __get_double_p(xm_o3_vr)
  C_xm_co2_vr = __get_double_p(xm_co2_vr)
  C_xm_n2o_vr = __get_double_p(xm_n2o_vr)
  C_xm_co_vr = __get_double_p(xm_co_vr)
  C_xm_ch4_vr = __get_double_p(xm_ch4_vr)
  C_zenith = c_double(zenith)
  C_albedo = c_double(albedo)
  
  solib.psrad_standalone_setup_multi(kbdim, klev, C_icldlyr, C_cldlyr_lwc, C_cldlyr_iwc, C_cldlyr_frc, C_zenith, C_albedo, C_pp_sfc, C_tk_sfc, C_hgt_fl_vr, C_pp_fl_vr, C_tk_fl_vr, C_xm_h2o_vr, C_xm_o3_vr, C_xm_co2_vr, C_xm_n2o_vr, C_xm_co_vr, C_xm_ch4_vr)

def advance_lrtm():
  "Execute long wave radiation routines. Results are then available through the 'get_lw_fluxes' function."
  solib.psrad_advance_lrtm()
def advance_srtm():
  "Execute short wave radiation routines. Results are then available through the 'get_sw_fluxes' function."
  solib.psrad_advance_srtm()

def get_lw_fluxes():
  """Returns the following arrays with the results of the long wave routines:
* htngrt(float64[klev]): Heating rate of total sky
* htngrt_clr(float64[klev]): Heating rate of clear sky
For these variables, position [0] is the value at the surface layer and position [klev-1] is at the TOA layer. NOTICE THIS IS THE OPPOSITE OF THE OTHER ROUTINE
* flxd_lw_vr(float64[klev+1]): Downward flux of total sky
* flxd_lw_clr_vr(float64[klev+1]): Downward flux of clear sky
* flxu_lw_vr(float64[klev+1]): Upward flux of total sky. 
* flxu_lw_clr_vr(float64[klev+1]): Upward flux of clear sky
For these variables, position [i] is the value at the bottom of layer [i], with 0 being the surface and [klev-1] being the last layer. Position [klev] is the TOA value."""
  from numpy import empty
  m = solib.psrad_get_kbdim()
  n = solib.psrad_get_klev()
  htngrt = empty([m,n], order="F", dtype="float64")
  htngrt_clr = empty([m,n], order="F", dtype="float64")
  n = n + 1
  flxd_lw_vr = empty([m,n], order="F", dtype="float64")
  flxd_lw_clr_vr = empty([m,n], order="F", dtype="float64")
  flxu_lw_vr = empty([m,n], order="F", dtype="float64")
  flxu_lw_clr_vr = empty([m,n], order="F", dtype="float64")
  C_htngrt = __get_double_p(htngrt)
  C_htngrt_clr = __get_double_p(htngrt_clr)
  C_flxd_lw_vr = __get_double_p(flxd_lw_vr)
  C_flxd_lw_clr_vr = __get_double_p(flxd_lw_clr_vr)
  C_flxu_lw_vr = __get_double_p(flxu_lw_vr)
  C_flxu_lw_clr_vr = __get_double_p(flxu_lw_clr_vr)
  solib.psrad_get_lw_fluxes(C_htngrt, C_htngrt_clr, C_flxd_lw_vr, C_flxd_lw_clr_vr, C_flxu_lw_vr, C_flxu_lw_clr_vr)
  return htngrt, htngrt_clr, flxd_lw_vr, flxd_lw_clr_vr, flxu_lw_vr, flxu_lw_clr_vr

def get_sw_fluxes():
  """Returns the following arrays with the results of the long wave routines:
* htngrt(float64[kbdim,klev]): Heating rate of total sky
* htngrt_clr(float64[kbdim,klev]): Heating rate of clear sky
For these variables, position [kbdim,klev-1] is the value at the surface layer and position [0] is at the TOA layer. NOTICE THIS IS THE OPPOSITE OF THE OTHER ROUTINE
* flxd_sw(float64[kbdim,klev+1]): Downward flux of total sky
* flxd_sw_clr(float64[kbdim,klev+1]): Downward flux of clear sky
* flxu_sw(float64[kbdim,klev+1]): Upward flux of total sky. 
* flxu_sw_clr(float64[kbdim,klev+1]): Upward flux of clear sky
For these variables, position [:,i] is the value at the top of layer [i], with 0 being at the TOA and [:,klev] being the surface.
* vis_frc_sfc(float64[kbdim]): Visible fraction of net surface radiation
* par_dn_sfc(float64[kbdim]): Photosynthetically active fraction of net surface radiation
* nir_dff_frc(float64[kbdim]): Diffuse near-infrared fraction
* vis_dff_frc(float64[kbdim]): Diffuse visible fraction
* par_dff_frc(float64[kbdim]): Diffuse photosynthetically active fraction"""
  from numpy import empty
  m = solib.psrad_get_kbdim()
  n = solib.psrad_get_klev()
  htngrt = empty([m,n], order="F", dtype="float64")
  htngrt_clr = empty([m,n], order="F", dtype="float64")
  n = n + 1
  flxd_sw = empty([m,n], order="F", dtype="float64")
  flxd_sw_clr = empty([m,n], order="F", dtype="float64")
  flxu_sw = empty([m,n], order="F", dtype="float64")
  flxu_sw_clr = empty([m,n], order="F", dtype="float64")
  vis_frc_sfc = empty(m, order="F", dtype="float64")
  par_dn_sfc = empty(m, order="F", dtype="float64")
  nir_dff_frc = empty(m, order="F", dtype="float64")
  vis_dff_frc = empty(m, order="F", dtype="float64")
  par_dff_frc = empty(m, order="F", dtype="float64")

  C_htngrt = __get_double_p(htngrt)
  C_htngrt_clr = __get_double_p(htngrt_clr)
  C_flxd_sw = __get_double_p(flxd_sw)
  C_flxd_sw_clr = __get_double_p(flxd_sw_clr)
  C_flxu_sw = __get_double_p(flxu_sw)
  C_flxu_sw_clr = __get_double_p(flxu_sw_clr)
  C_vis_frc_sfc = __get_double_p(vis_frc_sfc)
  C_par_dn_sfc = __get_double_p(par_dn_sfc)
  C_nir_dff_frc = __get_double_p(nir_dff_frc)
  C_vis_dff_frc = __get_double_p(vis_dff_frc)
  C_par_dff_frc = __get_double_p(par_dff_frc)

  solib.psrad_get_sw_fluxes(C_htngrt, C_htngrt_clr, C_flxd_sw, C_flxd_sw_clr, C_flxu_sw, C_flxu_sw_clr) 
  solib.psrad_get_sw_diags_old(C_vis_frc_sfc, C_par_dn_sfc, C_nir_dff_frc, C_vis_dff_frc, C_par_dff_frc)
  return htngrt, htngrt_clr, flxd_sw, flxd_sw_clr, flxu_sw, flxu_sw_clr, vis_frc_sfc, par_dn_sfc, nir_dff_frc, vis_dff_frc, par_dff_frc

