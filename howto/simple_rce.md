---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Our first simulation

Set up and run a radiative-convective equilibrium simulation

```{code-cell} ipython3
import matplotlib.pyplot as plt
from typhon import plots

import konrad


plev, phlev = konrad.utils.get_pressure_grids(1000e2, 1, 128)
atmosphere = konrad.atmosphere.Atmosphere(phlev)

# Plot the initial temperature profile (for comparison).
plt.style.use(plots.styles('typhon'))
fig, ax = plt.subplots()
plots.profile_p_log(atmosphere['plev'], atmosphere['T'][-1, :], label='Initial state')

# Initialize the setup for the radiative-convective equilibrium simulation.
rce = konrad.RCE(
    atmosphere,
    surface=konrad.surface.FixedTemperature(288.),
    timestep='12h',  # Set timestep in model time.
    max_duration='100d',  # Set maximum runtime.
)
rce.run()  # Start the simulation.

# Plot the equilibrium temperature profile.
plots.profile_p_log(atmosphere['plev'], atmosphere['T'][-1, :], label='RCE')
ax.legend()
```
