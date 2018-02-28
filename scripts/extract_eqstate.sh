#! /bin/bash
#
# Extract the last timestep of a netCDF file and store all atmospheric and
# surface porperties to a given new file.
#
# Usage:
#     $ ./extract_eqstate.sh full_results.nc last_timestep.nc
#
tmpfile=$(mktemp)  # Create temporary file for usage with ncks.

# Extract all atmosphere variables.
ncks -O -v phlev,plev,T,H2O,N2O,O3,CO2,CO,CH4,z,temperature "$1" "${tmpfile}"

cdo seltimestep,-1 "${tmpfile}" "$2"  # Extract last timestep.

# Append surface parameters from original netCDF file as `cdo` drops
# dimensionless variables when selecting a timestep.
ncks -A -v albedo,pressure,height "$1" "$2"

rm "${tmpfile}"  # Removed temporary file.
