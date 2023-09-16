#!/usr/bin/env python #
"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import matplotlib.pylab as plt

# -------------------------
# GVF_traj global variables
# -------------------------

# -90º rotation matrix
E = np.array([[0, 1],[-1, 0]]) 

# ------------------
# GVF_traj utilities
# ------------------

"""\
EMPTY
"""

# ---------------------
# GVF_traj common class
# ---------------------
class gvf_traj:
  def __init__(self):
    # Variables to draw the trajectory evaluation and vector field
    self.traj_points = [[]]
    self.mapgrad_pos = [[]]
    self.mapgrad_vec = [[]]

    self.R = np.eye(2,2)

  def draw(self, fig=None, ax=None, xlim=None, ylim=None, draw_field=True, alpha=0.2, ls="--", lw=1, width=0.0025):
    if fig == None:
      fig = plt.figure(dpi=100)
      ax = fig.subplots()
    elif ax == None:
      ax = fig.subplots()

    ax.plot(self.XYoff[0], self.XYoff[1], "+k", zorder=0)
    ax.plot(self.traj_points[0], self.traj_points[1], "k", ls=ls, lw=lw, zorder=0)

    if xlim:
      ax.set_xlim(xlim)
    if ylim:
      ax.set_ylim(ylim)

    if draw_field:
      field = ax.quiver(self.mapgrad_pos[:,0], self.mapgrad_pos[:,1], \
                        self.mapgrad_vec[:,0], self.mapgrad_vec[:,1], \
                        alpha=alpha, width=width)
      return fig, ax, field