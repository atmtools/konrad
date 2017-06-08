# -*- coding: utf-8 -*-
import sys
from distutils.core import setup
from setuptools import find_packages

from conrad import __version__

if not sys.version_info >= (3, 5, 1):
    sys.exit('Only support Python version >=3.5.1\n'
             'Found version is {}'.format(sys.version))

setup(
    name='conrad',
    author='Lukas Kluft',
    author_email='lukas.kluft@gmail.com',
    version=__version__,
    packages=find_packages(),
    license='MIT',
    description='Implementation of a radiative-convective equilibrium model.',
    classifiers=[
        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    include_package_data=True,
    install_requires=[
        'matplotlib>=2.0.0',
        'netcdf4>=1.2.7',
        'numba>=0.33.0',
        'numpy>=1.12.0',
        'scipy>=0.19.0',
        'typhon>=0.3.6',
        'xarray>=0.9.1',
    ],
)
