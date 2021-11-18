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
execution:
  timeout: 180
---

# Surface

```{code-cell} ipython3
import matplotlib.pyplot as plt
from typhon import plots

import konrad


plots.styles.use('typhon')
```

## Fixed surface temperature

The simplest representatio of a surface in `konrad` is the fixed surface temperature.
It is defined by an albedo and a prescribed temperature.

```{code-cell} ipython3
surface = konrad.surface.FixedTemperature(temperature=288., albedo=0.3)
```

When using a fixed surface temperature, the atmosphere component will converge to a consistent equilibrium quite fast (order of ~100 days)

```{code-cell} ipython3
plev, phlev = konrad.utils.get_pressure_grids(1000e2, 1, 128)
atmosphere = konrad.atmosphere.Atmosphere(phlev)

rce = konrad.RCE(
    atmosphere,
    surface=konrad.surface.FixedTemperature(temperature=288.),  # Run with a fixed surface temperature.
    timestep='12h',  # Set timestep in model time.
    max_duration='100d',  # Set maximum runtime.
)
rce.run()  # Start the simulation.
```

## Adjustable slab ocean

In addition, `konrad` provides a slab surface component. In contrast to the fixed surface temperature, the slab surface will adjust it's temperature depending on the net energy budget at the surface layer.
This energy consists of the net radiative fluxes as wel las an enthalpy sink that can be thought of as heat transport in the ocean layer (e.g. by ocean currents to the extra-tropics).
The value of the heat sink can be chosen to tune the energy budget. Here, we set it equal to the net energy balance at top-of-the-atmosphere from the previous fixed temperature run. This way, the model should be in equilibrium even if the surface temperature is allows to change.

```{code-cell} ipython3
slab = konrad.surface.SlabOcean(
    temperature=rce.surface["temperature"][-1],
    heat_sink=rce.radiation["toa"][-1],
)
```

Using the slab surface we can force our model with a doubling of CO2 and see the temperature slowly changing.

```{code-cell} ipython3
atmosphere["CO2"][:] *= 2

print("Initial temperature: ", slab["temperature"][0])

rce = konrad.RCE(
    atmosphere,
    surface=slab,  # Run with a fixed surface temperature.
    timestep='12h',  # Set timestep in model time.
    max_duration='300d',  # Set maximum runtime.
)
rce.run()  # Start the simulation.

print("Temperature after 300 days: ", slab["temperature"][0])
```
