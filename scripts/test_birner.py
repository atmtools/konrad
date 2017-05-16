# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""Perform radiative-equilibirum test simulations.

Note:
    This script is meant for debugging and testing. It should **not** be used
    as an example, as it might include functionality which is under
    development.
"""
import conrad
import matplotlib.pyplot as plt
import typhon


# Load the FASCOD atmosphere.
gf = typhon.arts.xml.load('data/tropical.xml')

# Refine original pressure grid.
p = typhon.math.nlogspace(1100e2, 0.1e2, 150)
gf.refine_grid(p, axis=1)

# Create an atmosphere model.
# a = conrad.atmosphere.AtmosphereFixedRH.from_atm_fields_compact(gf)
a = conrad.atmosphere.AtmosphereConUp.from_atm_fields_compact(gf)

# Create a sufrace model.
# s = conrad.surface.SurfaceAdjustableTemperature.from_atmosphere(a)
s = conrad.surface.SurfaceCoupled.from_atmosphere(a)

# Create a sufrace model.
r = conrad.radiation.PSRAD(atmosphere=a, surface=s)

# Combine atmosphere and surface model into an RCE framework.
rce = conrad.RCE(
    atmosphere=a,
    surface=s,
    radiation=r,
    delta=0.001,
    timestep=0.2,
    writeevery=1,
    max_iterations=800,
    outfile='results/test.nc'
    )

# The with block is not required for the model to run but prevents
# creating and removing of symlinks during each iteration.
with conrad.radiation.utils.PsradSymlinks():
    rce.run()  # Start simulation.

# Plot results
plt.style.use(typhon.plots.styles('typhon'))
fig, axes = plt.subplots(1, 3, sharey=True,
                         figsize=typhon.plots.figsize(10, portrait=False))
conrad.plots.plot_overview_p_log(
        data=rce.atmosphere,
        lw_htngrt=rce.heatingrates['lw_htngrt'][0, :],
        sw_htngrt=rce.heatingrates['sw_htngrt'][0, :],
        axes=axes,
        )
fig.suptitle('FASCOD tropical')

for ax in axes:
    ax.set_ylim(p.max(), p.min())
    ax.grid('on')

fig.savefig('plots/heatingrates.pdf')
