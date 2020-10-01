# -*- coding: utf-8 -*-
import sys
from os.path import (dirname, join)
from distutils.core import setup
from setuptools import find_packages


__version__ = open(join(dirname(__file__), 'konrad', 'VERSION')).read().strip()

with open('README.md') as f:
    long_description = f.read()

setup(
    name='konrad',
    author='The konrad developers',
    version=__version__,
    url='https://github.com/atmtools/konrad',
    download_url='https://github.com/atmtools/konrad/tarball/v' + __version__,
    packages=find_packages(),
    license='MIT',
    description='Implementation of a radiative-convective equilibrium model.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6',
    include_package_data=True,
    install_requires=[
        'matplotlib>=2.0.0',
        'netcdf4>=1.2.7',
        'numpy>=1.16.0',
        'scipy>=0.19.0',
        'typhon>=0.7.0',
        'xarray>=0.9.1',
        'climt==0.16.9',
        'sympl>=0.4.0',
    ],
    extras_require={
        'arts': [
            'pyarts',
        ],
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
        ],
        'tests': [
            'pytest',
        ],
    },
)
