# -*- coding: utf-8 -*-
"""Example control script for a simple RCE simulation."""
import konrad


# Create an atmosphere model from FASCOD climatology.
atmosphere = konrad.atmosphere.Atmosphere.from_xml(
    xmlfile='data/tropical-standard.xml')

# Refine the pressure grid to cover 200 vertical levels.
pgrid = konrad.utils.refined_pgrid(1000e2, 0.01e2, 200)
atmosphere = atmosphere.refine_plev(pgrid)

# Atmosphere composition according to RCEMIP.
atmosphere.tracegases_rcemip()

# Uncomment the following line to double the CO2 concentration.
# atmosphere['CO2'] *= 2

# Frame the setup for the radiative-convective equilibrium simulation.
rce = konrad.RCE(
    atmosphere,
    radiation=konrad.radiation.RRTMG(),  # Use RRTMG radiation scheme.
    timestep=0.5,  # Set timestep in days (12-hour timestep).
    max_iterations=712,  # Set maximum number of iterations (2 years).
    outfile='rce_simple.nc',  # Write output to netCDF file.
)

rce.run()  # Start the simulation.
