# conrad - Radiative-convective equilibrium framework

The goal of ``conrad`` is to provide a simple framework to run
radiative-convective equilibrium (RCE) simulations. It is build in an object
oriented structure to allow simple modifications of the model setup
(see [Structure](#structure)).

The package requires Python 3.5.1 or higher and a compiled version of the
radiaitve transfer model PSRAD (see [Requirements](#requirements)).
Further information about installation can be found in the section
[Installation](#installation).

# Requirements
## Python 3.5+
To use ``conrad`` you need Python 3.5.1 or higher. The recommended way to get
Python is through [Anaconda](https://www.continuum.io/downloads).
But of course, any other Python distribution is also working.

## PSRAD
The RCE simulations rely on external radiative-transfer code. Currently this
functionality is provided by the PSRAD radiation scheme. You need a compiled
version of this code in order to install and run ``conrad``.

A stable version is accessible through the internal subversion repository:
```bash
svn co \
https://arts.mi.uni-hamburg.de/svn/internal/browser/psrad/trunk psrad
```

Follow the instructions given in the repository to compile PSRAD on your
machine. A part of the installation process is to set some environment
variables. Thos are also needed in order to run ``conrad``:
```bash
source psrad_env.bash
```

# Installation
After installing PSRAD you can use ``pip`` to install the ``conrad`` package
to your Python distribution. On the top directory level, run:
```bash
pip install -e .
```

# Structure
