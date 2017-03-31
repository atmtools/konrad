# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
import conrad
import matplotlib.pyplot as plt
import typhon
from typhon.arts import xml


fascod_seasons = [
    'subarctic-winter',
    'subarctic-summer',
    'midlatitude-winter',
    'midlatitude-summer',
    'tropical',
]

for season in fascod_seasons:
    gf = xml.load('data/{}.xml'.format(season))

    # Refine original pressure grid to 200 levels.
    p = typhon.math.nlogspace(1100e2, 0.1e2, 75)
    gf.refine_grid(p, axis=1)

    data = conrad.utils.atmfield2pandas(gf)
    data['O3'] *= 0.01

    surface = conrad.surface.SurfaceAdjustableTemperature(
        temperature=data['T'].values[0],
        pressure=data['P'].values[0],
        )

    c = conrad.ConRad(
        sounding=data,
        surface=surface,
        dt=1,
        max_iterations=500,
        fix_rel_humidity=True,
        outfile='results/{}.nc'.format(season)
        )
    c.run()

# Plot final result.
# fig, axes = plt.subplots(1, 3, sharey=True, figsize=(8, 6))
# c.plot_overview_z(axes)
# for ax in axes:
#     ax.set_ylim(0, 30)

# plt.show()
