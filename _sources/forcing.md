---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Radiative forcing

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from typhon import plots

import konrad


plots.styles.use()
```

## Reference climate

In a first step, we equilibrate our model in a reference climate state.

```{code-cell} ipython3
phlev = konrad.utils.get_quadratic_pgrid(1000e2, 10, 128)
atmosphere = konrad.atmosphere.Atmosphere(phlev)
atmosphere["CO2"][:] = 348e-6 # Set reference CO2 concentration

# Calculate reference OLR.
spinup = konrad.RCE(atmosphere, timestep='24h', max_duration='150d')
spinup.run()
```

After the model has converged, we store the outgoing-longwave radiation at the top-of-the-atmosphere.
This value is our reference against which we will later compute the radiative forcing.

```{code-cell} ipython3
olr_ref = spinup.radiation["lw_flxu"][-1, -1]
```

## Instantanoues forcing

The instantanoues forcing is the change in OLR that is induced by a doubling of the CO2 concentration only.
In this defintion, no other changes to the atmospheric state - in particular the temperature - are allowed.

```{code-cell} ipython3
# Calculate OLR at perturbed atmospheric state.
atmosphere["CO2"][:] *= 2  # double the CO2 concentration
spinup.radiation.update_heatingrates(atmosphere)

instant_forcing = -(spinup.radiation["lw_flxu"][-1, -1] - olr_ref)
print(f"Instantanoues forcing: {instant_forcing:.2f} W/m^2")
```

## Effective forcing

The effective forcing includes the so called "stratospheric adjustment". Due to a significant cooling of the stratosphere, the radiative forcing at the top-of-the-atmosphere is increased.
The effective forcing is a better description of the radiative imbalance that actually forces the troposphere.

```{code-cell} ipython3
perturbed = konrad.RCE(atmosphere, timestep='24h',max_duration='150d')
perturbed.run()

effective_forcing = -(perturbed.radiation["lw_flxu"][-1, -1] - olr_ref)
print(f"Effective forcing: {effective_forcing:.2f} W/m^2")
```
