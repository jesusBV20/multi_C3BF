#!/usr/bin/env python #
"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import matplotlib.pylab as plt

from .gvf_traj import *

# ----------------------------------------------------------------------------
# The computation are vectorized to be performed with N agents.
# -------------------------------------------------------------
# -> p      (N x 2)
# -> XYoff  (1 x 2)
# -> R is a rotation matrix (its inverse is equivalent to its transpose)
# ----------------------------------------------------------------------------

class gvf_eight(gvf_traj):
  def __init__(self, XYoff, r1, r2):
    super().__init__()

    # Eight parameters
    self.XYoff = XYoff
    self.r1, self.r2 = r1, r2

    # Get the rajectory points
    self.traj_points = self.param_points()

  """\
  Function to cumpute the trajectory points
  """
  def param_points(self, pts = 100):
    theta = np.linspace(0, 2*np.pi, pts)
    r = self.r1 + self.r2 * np.cos(theta)**2

    # Transform from polar to cartesian coordenates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return [x, y]

  """\
  Phi(p)
  """
  def phi(self, p):
    x, y = p[:,0], p[:,1]
    r = np.sqrt((x**2 + y**2))

    return r - self.r1 - self.r2 * (x/r)**2

  """\
  Phi gradiant
  """
  def grad_phi(self, p):
    x, y = p[:,0], p[:,1]
    r = np.sqrt((x**2 + y**2))

    return np.array([x/r - 2*self.r2*(x/r)*(1/r - x**2/r**3), \
                     y/r - 2*self.r2*(x/r)*(-x*y/r**3)]).T

  """\
  Hessian
  """
  def hess_phi(self, p):
    x, y = p[:,0], p[:,1]
    r = np.sqrt((x**2 + y**2))

    H = np.zeros((p.shape[0],2,2))
    H[:,0,0] = (1/r - x**2/r**3) - 2*self.r2*(1/r**2 - x**2/r**5 - 3*x**2/r**4 + x**4/r**9)
    H[:,0,1] = (-x*y/r**3) - 2*self.r2*(-x*y/r**5 + x**3*y/r**9)
    H[:,1,0] = (-x*y/r**3) - 2*self.r2*(-2*x*y/r**4 + x**3 * y / r**9)
    H[:,1,1] = (1/r - y**2/r**3) - 2*self.r2*(-x**2/r**4 + x**2 * y**2 / r**9)
    return H

  """\
  Funtion to generate the vector field to be plotted
  """
  def vector_field(self, XYoff, area, s, ke, kr = 1, pts = 30):
    x_lin = np.linspace(XYoff[0] - 0.5*np.sqrt(area), \
                        XYoff[0] + 0.5*np.sqrt(area), pts)
    y_lin = np.linspace(XYoff[1] - 0.5*np.sqrt(area), \
                        XYoff[1] + 0.5*np.sqrt(area), pts)
    mapgrad_X, mapgrad_Y = np.meshgrid(x_lin, y_lin)
    mapgrad_X = np.reshape(mapgrad_X, -1)
    mapgrad_Y = np.reshape(mapgrad_Y, -1)
    self.mapgrad_pos = np.array([mapgrad_X, mapgrad_Y]).T

    w = self.XYoff * np.ones([self.mapgrad_pos.shape[0],1])
    pel = (self.mapgrad_pos - w) @ self.R

    n = self.grad_phi(self.mapgrad_pos)
    t = s*n @ E.T

    e = self.phi(self.mapgrad_pos)[:,None]

    self.mapgrad_vec = t - ke*e*n

    norm = np.sqrt(self.mapgrad_vec[:,0]**2 + self.mapgrad_vec[:,1]**2)[:,None]
    self.mapgrad_vec = self.mapgrad_vec / norm

