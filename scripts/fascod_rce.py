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

season = 'subarctic-winter'
gf = xml.load('data/{}.xml'.format(season))
data = conrad.utils.atmfield2pandas(gf)

c = conrad.ConRad(
    sounding=data,
    adjust_vmr=True,
    max_iterations=1000,
    plot_iterations=False,
)

c.run()

# Plot final result.
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(8, 6))
c.plot_overview_z(fig)
plt.show()
