#!/usr/bin/env python #
"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import matplotlib.pylab as plt

from .gvf_traj import *

# ----------------------------------------------------------------------------
# Todas los cálculos están vectorizados para poder realizarlos sobre N agentes
#
# -> p es una matriz     N x 2
# -> XYoff es un vector  1 x 2
# -> R es una matriz rotación (su inversa equivale a su transpuesta)
# ----------------

class gvf_ellipse(gvf_traj):
  def __init__(self, XYoff, alpha, a, b):
    super().__init__()

    # Parámetros de la elipse
    self.XYoff = XYoff
    self.alpha = alpha
    self.a, self.b = a, b

    self.cosa, self.sina = np.cos(alpha), np.sin(alpha)
    self.R = np.array([[self.cosa, self.sina], [-self.sina, self.cosa]])

    # Variables para dibujar la trayectoria y el campo
    self.traj_points = self.param_points()

    # Phi hessian
    self.hess = np.zeros((2,2))
    self.hess[0,0] = 2 * (self.cosa**2 / self.a**2 + self.sina**2 / self.b**2)
    self.hess[0,1] = 2 * self.sina * self.cosa * (1 / self.b**2 - 1 / self.a**2)
    self.hess[1,0] = self.hess[0,0]
    self.hess[1,1] = 2 * (self.sina**2 / self.a**2 + self.cosa**2 / self.b**2)

  # Puntos para pintar la trayectoria
  def param_points(self, pts = 100):
    t = np.linspace(0, 2*np.pi, pts)
    x = self.XYoff[0] + self.a * np.cos(-self.alpha) * np.cos(t) \
                      - self.b * np.sin(-self.alpha) * np.sin(t)
    y = self.XYoff[1] + self.a * np.sin(-self.alpha) * np.cos(t) \
                      + self.b * np.cos(-self.alpha) * np.sin(t)
    return [x, y]

  # Phi
  def phi(self, p):
    w = self.XYoff * np.ones([p.shape[0],1])
    pel = (p - w) @ self.R
    return (pel[:,0]/self.a)**2 + (pel[:,1]/self.b)**2 - 1

  # Phi gradiant
  def grad_phi(self, p):
    w = self.XYoff * np.ones([p.shape[0],1])
    pel = (p - w) @ self.R
    return 2 * pel / [self.a**2, self.b**2] @ self.R.T

  # Generamos el campo de vectores a dibujar
  # pn es una matriz N x 2
  def vector_field(self, XYoff, area, s, ke, kb = 0, kr = 1, pn = None, n_focus = None, pts = 30):
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

    if kb != 0:
      # Añadimos una interacción de repulsión electroestática basa en iones
      for i in range(pn.shape[0]):
        if i!=n_focus:
          r_vec  = self.mapgrad_pos - pn[i,:]
          r_norm = np.sqrt(r_vec[:,0]**2 + r_vec[:,1]**2)[:,None]
          self.mapgrad_vec += kb / r_norm**(1 + kr) * r_vec

    norm = np.sqrt(self.mapgrad_vec[:,0]**2 + self.mapgrad_vec[:,1]**2)[:,None]
    self.mapgrad_vec = self.mapgrad_vec / norm

  # Se manda la información necesaria para el controlador
  def info(self, p):
    return self.phi(p), self.grad_phi(p), self.hess

# ----------------------------------------------------------------------------

"""
Función para dibujar el campo vectorial y la trayectoria.
"""
def draw(gvf_ell, fig=None, ax=None, xlim=None, ylim=None, draw_field=True):
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
                      alpha=0.5)
    return fig, ax, field

