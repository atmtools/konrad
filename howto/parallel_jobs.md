---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Running in parallel

One of the advantages of simple models like `konrad` is their low computational
cost. This allows the user to explore a wide range of parameters in ensemble
simulations.

The python standard library includes the `multiprocessing` package, which
allows to run these ensembles in parallel.

% Don't run the code block because of issues with Jupyter and multiprocessing!
```{code-block} ipython3
:emphasize-lines: 26,27
import itertools
import multiprocessing

import konrad


def run_rce(Ts=300., RH=0.8):
    """Run an RCE for a given combination of Ts and RH."""
    rce = konrad.RCE(
        atmosphere=konrad.atmosphere.Atmosphere(konrad.utils.get_quadratic_pgrid(num=64)),
        surface=konrad.surface.FixedTemperature(temperature=Ts), 
        humidity=konrad.humidity.FixedRH(konrad.humidity.VerticallyUniform(RH)), 
    )
    rce.run()

    return rce


# Define set of parameters
Tss = [273, 288, 294, 300, 305]
RHs = [.1, .25, .5, .75, .9]

rce_configs = list(itertools.product(Tss, RHs))

# Run processes in parallel
with multiprocessing.Pool(processes=16) as pool:
    rce_results = pool.starmap(run_rce, rce_configs)

# Combine results in a dict.
# (In python, tuples are immutable and can be used as dict-keys directly.)
rces = dict(zip(rce_configs, rce_results))

# Print results
for (t, rh), rce in rces.items():
    print(f"T_s: {t}K", f"RH {rh:.0%}", f"OLR: {rce.radiation['lw_flxu'][-1, -1]:.2f}")

```
