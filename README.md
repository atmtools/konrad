[![PyPI version](https://badge.fury.io/py/konrad.svg)](https://badge.fury.io/py/konrad)
[![Test](https://github.com/atmtools/konrad/workflows/Test/badge.svg?branch=main)](https://github.com/atmtools/konrad/commits/main)
[![Documentation Status](https://readthedocs.org/projects/konrad/badge/?version=latest)](https://konrad.readthedocs.io/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1313687.svg)](https://doi.org/10.5281/zenodo.1313687)

# ![Logo](howto/images/konrad-logo_64.png) konrad

``konrad`` is a one-dimensional radiative-convective equilibrium (RCE) model.
It is build in an object oriented structure to allow simple modifications of
the model setup.

You can find various tutorials that illustrate the usage of ``konrad`` in our
["How to konrad"](https://atmtools.github.io/konrad) Jupyter book.

## Requirements
``konrad`` requires Python 3.6 or higher. The recommended way to get
Python is through [Anaconda](https://www.continuum.io/downloads).
But of course, any other Python distribution is also working.

## Install stable release
You can install the latest stable version of ``konrad`` using ``pip``:
```bash
python -m pip install konrad
```

Konrad depends on the [CliMT](https://github.com/CliMT/climt) package.
CliMT handles a variety of underlying FORTRAN code and provides precompiled
binary wheels for some Python versions and operating systems.

However (for Python >3.7) the FORTRAN libraries need to be compiled locally.
In this case, you need to specify a C compiler, a FORTRAN compiler, and the
target architecture using the corresponding environment variables:
```bash
CC=gcc FC=gfortran TARGET=HASWELL python -m pip install konrad
```

### macOS
On macOS, you may need to install the GCC compiler suite beforehand:
```bash
# Install GCC 11 and set it as C and Fortran compiler.
brew install gcc@11
CC=gcc-11 FC=gfortran-11

# Set the target architecture (different for Apple M1 [arm64]).
[[ $(uname -p) == arm64 ]] && TARGET=ARMV8 || TARGET=HASWELL

# Install a trimmed down version of CliMT that ships RRTMG only.
export CC FC TARGET
python -m pip install git+https://github.com/atmtools/climt@rrtmg-only

# Install konrad itself
python -m pip install konrad
```
