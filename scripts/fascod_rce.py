# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
import conrad
import matplotlib.pyplot as plt
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
    data = conrad.utils.atmfield2pandas(gf)

    c = conrad.ConRad( sounding=data, outfile='results/{}.nc'.format(season))
    c.run()

# # Plot final result.
# fig, axes = plt.subplots(1, 3, sharey=True, figsize=(8, 6))
# c.plot_overview_z(axes)
# for ax in axes:
#     ax.set_ylim(0, 30)

# fig, ax = plt.subplots(figsize=(2.5, 6))
# c.plot_sounding_z('T', ax)
# ax.set_ylim(0, 30)

plt.show()
