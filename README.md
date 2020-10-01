[![PyPI version](https://badge.fury.io/py/konrad.svg)](https://badge.fury.io/py/konrad)
[![Test](https://github.com/atmtools/konrad/workflows/Test/badge.svg?branch=master)](https://github.com/atmtools/konrad/commits/master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1313687.svg)](https://doi.org/10.5281/zenodo.1313687)
[![Documentation Status](https://readthedocs.org/projects/konrad/badge/?version=latest)](https://konrad.readthedocs.io/en/latest/?badge=latest)

# konrad - Radiative-convective equilibrium framework

The goal of ``konrad`` is to provide a simple framework to run
radiative-convective equilibrium (RCE) simulations. It is build in an object
oriented structure to allow simple modifications of the model setup.

## Requirements
``konrad`` requires Python 3.6 or higher. The recommended way to get
Python is through [Anaconda](https://www.continuum.io/downloads).
But of course, any other Python distribution is also working.

## Install stable release
You can install the latest stable version of ``konrad`` using ``pip``:
```bash
pip install konrad
```

Konrad depends on the [CliMT](https://github.com/CliMT/climt) package.
CliMT handles a variety of underlying FORTRAN code and provides precompiled
binary wheels for some Python versions and operating systems.

### Python >3.7
However, for Python >3.7 the FORTRAN libraries need to be compiled locally.
In this case, you need to specify a C compiler, a FORTRAN compiler, and the
target architecture using the corresponding environment variables:
```bash
CC=gcc FC=gfortran TARGET=HASWELL pip install konrad
```

### macOS
On macOS, you may need to install the GCC compiler suite beforehand:
```bash
brew install gcc@10
CC=gcc-10 FC=gfortran-10 TARGET=HASWELL pip install konrad
```
