# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""Calculate Jacobians for heatingrates to determine contribution per species.
"""
import numpy as np
import typhon

import conrad


# Load the FASCOD atmosphere.
gf = typhon.arts.xml.load('data/tropical.xml')
gf = gf.extract_slice(slice(1, None), axis=1)  # omit bottom level.

# Refine original pressure grid.
p = typhon.math.nlogspace(1013e2, 0.1e2, 200)
gf.refine_grid(p, axis=1)

# Create an atmosphere model.
a = conrad.atmosphere.AtmosphereFixedRH.from_atm_fields_compact(gf)
rh_profile = conrad.utils.create_relative_humidity_profile(p, 0.75)
a.relative_humidity = rh_profile

# Create a sufrace model.
s = conrad.surface.SurfaceAdjustableTemperature.from_atmosphere(a)

# Create a sufrace model.
r = conrad.radiation.PSRAD(atmosphere=a, surface=s)

# Combine atmosphere and surface model into an RCE framework.
rce = conrad.RCE(
    atmosphere=a,
    surface=s,
    radiation=r,
    delta=0.001,
    timestep=0.0625,
    max_iterations=1,
    )

with conrad.radiation.utils.PsradSymlinks():
    rce.run()  # Start simulation.
    hr_ref = rce.heatingrates['net_htngrt'][0, :].values.copy()

    K = np.zeros(p.size)  # pre-allocate Jacobian matrix.
    for layer in range(p.size):
        # Reset atmosphere.
        a = conrad.atmosphere.AtmosphereFixedRH.from_atm_fields_compact(gf)
        a.relative_humidity = rh_profile
        a_ref = a['O3'][0, layer].values.copy()
        factor = 1.1
        a['O3'][0, layer] *= factor  # Perturbate humidity profile.

        rce = conrad.RCE(
            atmosphere=a,
            surface=s,
            radiation=r,
            delta=0.001,
            timestep=0.0625,
            max_iterations=1,
            )
        rce.run()  # Start simulation.

        divisor = (rce.heatingrates['net_htngrt'][0, layer] - hr_ref[layer])
        divider = (a['O3'][0, layer] - a_ref)
        K[layer] = divisor / divider

    typhon.arts.xml.save(K, "results/jacobian.xml")
    typhon.arts.xml.save(p, "results/plev.xml")
