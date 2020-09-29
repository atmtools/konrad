Humidity
========

Wrapper classes
---------------

.. automodule:: konrad.humidity

.. autosummary::
   :toctree: _autosummary

   FixedVMR
   FixedRH

Relative humidity
-----------------

.. automodule:: konrad.humidity.relative_humidity

.. autosummary::
   :toctree: _autosummary

   RelativeHumidityModel
   CacheFromAtmosphere
   HeightConstant
   VerticallyUniform
   ConstantFreezingLevel
   FixedUTH
   CoupledUTH
   CshapeConstant
   CshapeDecrease
   Manabe67
   Cess76
   Romps14
   PolynomialCshapedRH
   PerturbProfile

Stratospheric coupling
----------------------

.. automodule:: konrad.humidity.stratosphere

.. autosummary::
   :toctree: _autosummary

   StratosphereCoupler
   ColdPointCoupling
   NonIncreasing
   FixedStratosphericVMR
   MinimumStratosphericVMR
