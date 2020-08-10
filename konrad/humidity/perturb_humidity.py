"""Different functions to add a perturbation to the relative humidity profile"""
import abc

import numpy as np

from konrad.component import Component
from konrad.physics import vmr2relative_humidity
from konrad.humidity.relative_humidity import *
from konrad.utils import gaussian

class PerturbHumidityModel(Component, metaclass=abc.ABCMeta):
    def __call__(self, atmosphere, **kwargs):
         """Return the perturbed vertical distribution of relative humidity.

        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere: Atmosphere component.
            **kwargs: Arbitrary number of additional arguments,
                depending on the actual implementation.

        Returns:
            ndarray: Relative humidity profile.
        """


class DiracPerturbation_FixedP(PerturbHumidityModel):
    """ Increase or decrease Relative humidity at one grid point, at a given pressure. """
    def __init__(self, base_profile = HeightConstant(), dirac_plev = 500e2, dirac_intensity = 0.1):
        """
        Parameters:
            base_profile (konrad.relative_humidity model): initial profile on which we will add the perturbation.
            dirac_plev (float): Pressure level of the perturbation in [Pa].
            dirac_intensity (float): Change in RH at the pressure level of the perturbation, positive or negative.
        """
        
        self._base_profile = base_profile
        self.dirac_plev = dirac_plev

        if dirac_intensity > 1 : #If intensity given in percents
            dirac_intensity /= 100

        self.dirac_intensity = dirac_intensity

    def __call__(self, atmosphere, **kwargs):
        """
        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): The atmosphere component.
            
        Returns:
            ndarray: The relative humdity profile.
        """

        plev = atmosphere["plev"]
        
        rh_profile = self._base_profile(atmosphere).copy()
        dirac_idx = np.abs(plev - self.dirac_plev).argmin()
        rh_profile[dirac_idx] += self.dirac_intensity

        # Security
        if rh_profile[dirac_idx]  > 1 : rh_profile[dirac_idx] = 1;
        if rh_profile[dirac_idx] < 0 : rh_profile[dirac_idx] = 0;

        return rh_profile

    
class DiracPerturbation_FixedT(PerturbHumidityModel):
    """ Increase or decrease Relative humidity at one grid point, at a given pressure. """
    def __init__(self, base_profile = HeightConstant(), dirac_temp = 273, dirac_intensity = 0.1):
        """
        Parameters:
            base_profile (konrad.relative_humidity model): initial profile on which we will add the perturbation.
            dirac_temp (float): Temperature of the perturbation in [K].
            dirac_intensity (float): Change in RH at the pressure level of the perturbation, positive or negative.
        """
        
        self._base_profile = base_profile
        self.dirac_temp = dirac_temp

        if dirac_intensity > 1 : #If intensity given in percents
            dirac_intensity /= 100

        self.dirac_intensity = dirac_intensity

    def __call__(self, atmosphere, **kwargs):
        """
        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): The atmosphere component.
            
        Returns:
            ndarray: The relative humdity profile.
        """

        T = atmosphere["T"][-1]
        
        rh_profile = self._base_profile(atmosphere).copy()
        dirac_idx = np.abs(T - self.dirac_temp).argmin()
        rh_profile[dirac_idx] += self.dirac_intensity

        # Security
        if rh_profile[dirac_idx]  > 1 : rh_profile[dirac_idx] = 1;
        if rh_profile[dirac_idx] < 0 : rh_profile[dirac_idx] = 0;

        return rh_profile

class SquarePerturbation_FixedP(PerturbHumidityModel):
    """ Uniform increase or decrease Relative humidity in a given zone. """
    def __init__(self, base_profile = HeightConstant(), center_plev = 500e2, square_width = 100e2, intensity = 0.1):
        """
        Parameters:
            base_profile (konrad.relative_humidity model): initial profile on which we will add the perturbation.
            center_plev (float): Pressure of the center of the square perturbation in [Pa].
            square_width (float): Span of the perturbation over the profile in [Pa].
            intensity (float): Change in RH where the profile is perturbed, positive or negative.
        """
        
        self._base_profile = base_profile
        self.center_plev = center_plev
        self.square_width = square_width

        if intensity > 1 : #If intensity given in percents
            intensity /= 100

        self.intensity = intensity

    def __call__(self, atmosphere, **kwargs):
        """
        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): The atmosphere component.
            
        Returns:
            ndarray: The relative humdity profile.
        """

        plev = atmosphere["plev"]
        
        rh_profile = self._base_profile(atmosphere).copy()
        low_idx = np.abs(plev - (self.center_plev + self.square_width/2)).argmin()
        high_idx = np.abs(plev - (self.center_plev - self.square_width/2)).argmin()
        rh_profile[low_idx:high_idx] += self.intensity

        print(low_idx, high_idx)
        
        return rh_profile
    
class SquarePerturbation_FixedT(PerturbHumidityModel):
    """ Uniform increase or decrease Relative humidity in a given zone. """
    def __init__(self, base_profile = HeightConstant(), center_temp = 273, square_width = 100e2, intensity = 0.1):
        """
        Parameters:
            base_profile (konrad.relative_humidity model): initial profile on which we will add the perturbation.
            center_temp (float): Temperature of the center of the square perturbation in [K].
            square_width (float): Span of the perturbation over the profile in [Pa].
            intensity (float): Change in RH where the profile is perturbed, positive or negative.
        """
        
        self._base_profile = base_profile
        self.center_temp = center_temp
        self.square_width = square_width

        if intensity > 1 : #If intensity given in percents
            intensity /= 100

        self.intensity = intensity

    def __call__(self, atmosphere, **kwargs):
        """
        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): The atmosphere component.
            
        Returns:
            ndarray: The relative humdity profile.
        """

        T = atmosphere["T"]
        plev = atmosphere["plev"]
        
        rh_profile = self._base_profile(atmosphere).copy()
        center_idx =  np.abs(T - self.center_temp).argmin()
        center_plev = plev[center_idx]
        low_idx = np.abs(plev - (center_plev + self.square_width/2)).argmin()
        high_idx = np.abs(plev - (center_plev - self.square_width/2)).argmin()
        rh_profile[low_idx:high_idx] += self.intensity
        
        return rh_profile

class GaussianPerturbation_FixedP(PerturbHumidityModel):
    """ Increase or decrease Relative humidity at with a gaussian function, at a given pressure. """
    def __init__(self, base_profile = HeightConstant(), center_plev = 500e2, std = 20e2, intensity = 0.1):
        """
        Parameters:
            base_profile (konrad.relative_humidity model): initial profile on which we will add the perturbation.
            center_plev (float): Pressure of the center of the square perturbation in [Pa].
            std (float): standard devation of the gaussian perturbation in [Pa].
            intensity (float): Change in RH where the profile is perturbed, positive or negative.
        """
        
        self._base_profile = base_profile
        self.center_plev = center_plev
        self.std = std

        if intensity > 1 : #If intensity given in percents
            intensity /= 100

        self.intensity = intensity

    def g(x, μ, σ) :
        """
        Returns the value of the gaussian function defined by mean and std at point x
        """ 
        X = (x-μ)**2 / (2*σ**2)
        return np.exp(-X)

    def __call__(self, atmosphere, **kwargs):
        """
        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): The atmosphere component.
            
        Returns:
            ndarray: The relative humdity profile.
        """

        plev = atmosphere["plev"]
        
        rh_profile = self._base_profile(atmosphere).copy()
        G = gaussian(plev, self.center_plev, self.std) # Gaussian profile

        # Compute boundary of the perturbation
        p_low = self.center_plev + 3*self.std
        idx_low = np.abs(plev - p_low).argmin()
        p_high = self.center_plev - 3*self.std
        idx_high = np.abs(plev - p_high).argmin()

        rh_profile[idx_low:idx_high] = rh_profile[idx_low:idx_high] + G[idx_low:idx_high]/np.max(G) * self.intensity

        return rh_profile

    
class GaussianPerturbation_FixedT(PerturbHumidityModel):
    """ Increase or decrease Relative humidity at with a gaussian function, at a given pressure. """
    def __init__(self, base_profile = HeightConstant(), center_temp = 273, std = 20e2, intensity = 0.1):
        """
        Parameters:
            base_profile (konrad.relative_humidity model): initial profile on which we will add the perturbation.
            center_plev (float): Pressure of the center of the square perturbation in [Pa].
            std (float): standard devation of the gaussian perturbation in [Pa].
            intensity (float): Change in RH where the profile is perturbed, positive or negative.
        """
        
        self._base_profile = base_profile
        self.center_temp = center_temp
        self.std = std

        if intensity > 1 : #If intensity given in percents
            intensity /= 100

        self.intensity = intensity

    def g(x, μ, σ) :
        """
        Returns the value of the gaussian function defined by mean and std at point x
        """ 
        X = (x-μ)**2 / (2*σ**2)
        return np.exp(-X)

    def __call__(self, atmosphere, **kwargs):
        """
        Parameters:
            atmosphere (konrad.atmosphere.Atmosphere): The atmosphere component.
            
        Returns:
            ndarray: The relative humdity profile.
        """
        
        T = atmosphere["T"]
        plev = atmosphere["plev"]
        
        rh_profile = self._base_profile(atmosphere).copy()
        center_idx =  np.abs(T - self.center_temp).argmin()
        center_plev = plev[center_idx]
        
        rh_profile = self._base_profile(atmosphere).copy()
        G = gaussian(plev, center_plev, self.std) # Gaussian profile

        # Compute boundary of the perturbation
        p_low = center_plev + 3*self.std
        idx_low = np.abs(plev - p_low).argmin()
        p_high = center_plev - 3*self.std
        idx_high = np.abs(plev - p_high).argmin()

        rh_profile[idx_low:idx_high] = rh_profile[idx_low:idx_high] + G[idx_low:idx_high]/np.max(G) * self.intensity

        return rh_profile
