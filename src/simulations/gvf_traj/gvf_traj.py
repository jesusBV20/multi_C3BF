#!/usr/bin/env python #
"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import matplotlib.pylab as plt

# -------------------------
# GVF_traj global variables
# -------------------------

E = np.array([[0, 1],[-1, 0]])

# ------------------
# GVF_traj utilities
# ------------------

"""
Función para dibujar el campo vectorial y la trayectoria.
"""
def draw(gvf_ell, fig=None, ax=None, xlim=None, ylim=None, draw_field=True, alpha=0.2):
  if fig == None:
    fig = plt.figure(dpi=100)
    ax = fig.subplots()
  elif ax == None:
    ax = fig.subplots()

  ax.plot(gvf_ell.XYoff[0], gvf_ell.XYoff[1], "+k", zorder=0)
  ax.plot(gvf_ell.traj_points[0], gvf_ell.traj_points[1], "k--", zorder=0)

  if xlim:
    ax.set_xlim(xlim)
  if ylim:
    ax.set_ylim(ylim)

  if draw_field:
    field = ax.quiver(gvf_ell.mapgrad_pos[:,0], gvf_ell.mapgrad_pos[:,1], \
                      gvf_ell.mapgrad_vec[:,0], gvf_ell.mapgrad_vec[:,1], \
                      alpha=alpha)
    return fig, ax, field

# ---------------
# GVF_traj class ##TODO: Common methods here
# ---------------
class gvf_traj:
  def __init__(self):
    # Variables para dibujar la trayectoria y el campo
    self.traj_points = [[]]
    self.mapgrad_pos = [[]]
    self.mapgrad_vec = [[]]