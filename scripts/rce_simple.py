# -*- coding: utf-8 -*-
"""Example control script for a simple RCE simulation."""
import typhon
import conrad


# Create an atmosphere model from FASCOD climatology.
gf = typhon.arts.xml.load('data/tropical-standard.xml')
atmosphere = conrad.atmosphere.Atmosphere.from_atm_fields_compact(gf)

# Refine the pressure grid to cover 200 vertical levels.
pgrid = conrad.utils.refined_pgrid(1013e2, 0.01e2, 200)
atmosphere = atmosphere.refine_plev(pgrid)

# Frame the setup for the radiative-convective equilibirum simulation.
rce = conrad.RCE(
    atmosphere,
    timestep=0.25,
    outfile='results/rce_simple.nc',
)

rce.run()  # Start simulation.
