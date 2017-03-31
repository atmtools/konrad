# Installation path for psrad
export PSRAD_PATH='/scratch/uni/u237/users/lkluft/icon-aes/psrad'

# Environment settings
export F90="gfortran"  # FORTRAN compiler
export HDF5ROOT="/sw/squeeze-x64/hdf5-1.8.8"
export NETCDFROOT="/sw/squeeze-x64/netcdf-4.1.3-gccsys"
export NETCDFFROOT="/sw/squeeze-x64/netcdf-4.1.3-gccsys"
export LD_LIBRARY_PATH="${HDF5ROOT}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NETCDFROOT}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PSRAD_PATH}"

# Include installation path to PATH (needed for bash).
export PATH="${PATH}:${PSRAD_PATH}"
