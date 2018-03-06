# Installation path for psrad
export PSRAD_PATH='/work/um0878/users/lkluft/icon-aes/psrad'

# Environment settings
export F90="gfortran"  # FORTRAN compiler
# export HDF5ROOT="/sw/rhel6-x64/hdf5/hdf5-1.8.14-threadsafe-gcc48"
export HDF5ROOT="/sw/rhel6-x64/hdf5/hdf5-1.8.16-gcc48"
export NETCDFROOT="/sw/rhel6-x64/netcdf/netcdf_c-4.3.2-gcc48"
export NETCDFFROOT="/sw/rhel6-x64/netcdf/netcdf_fortran-4.4.2-gcc48"
export LD_LIBRARY_PATH="${HDF5ROOT}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NETCDFROOT}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NETCDFFROOT}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PSRAD_PATH}"

# Disable HDF5 version check due to collision with anaconda HDF5 headers.
export HDF5_DISABLE_VERSION_CHECK=1

# Include installation path to PATH (needed for bash).
export PATH="${PATH}:${PSRAD_PATH}"
