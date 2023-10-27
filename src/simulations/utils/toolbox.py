#!/usr/bin/env python #
"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import os
import numpy as np
import matplotlib.pylab as plt

FIGSIZE = [16, 9]
RES = 1920 # 720p

from matplotlib.path import Path
import matplotlib.patches as patches

# Cake color palette (https://g.co/kgs/4A3q4T)
cake_b = "#d9daff"
cake_r = "#ffdbd9"
cake_y = "#fffbd9"
cake_g = "#dcffd9"

"""\
Unicycle_patch (Héctor García de Marina) -
"""
def unicycle_patch(XY, yaw, color, size=1, lw=0.5):
    # XY is a list [X, Y]
    Rot = np.array([[np.cos(yaw), np.sin(yaw)],[-np.sin(yaw), np.cos(yaw)]])

    apex = 45*np.pi/180 # 30 degrees apex angle
    b = np.sqrt(1) / np.sin(apex)
    a = b*np.sin(apex/2)
    h = b*np.cos(apex/2)

    z1 = size*np.array([a/2, -h*0.3])
    z2 = size*np.array([-a/2, -h*0.3])
    z3 = size*np.array([0, h*0.6])

    z1 = Rot.dot(z1)
    z2 = Rot.dot(z2)
    z3 = Rot.dot(z3)

    verts = [(XY[0]+z1[1], XY[1]+z1[0]), \
             (XY[0]+z2[1], XY[1]+z2[0]), \
             (XY[0]+z3[1], XY[1]+z3[0]), \
             (0, 0)]

    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(verts, codes)

    return patches.PathPatch(path, fc=color, lw=lw)

"""\
Function to format the data axis following the same rule -
"""
def fmt_data_axis(ax, ylabel = "", xlabel = "", title = "",
                  xlim = None, ylim = None, invy = True, d=2):
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)

  if xlim is not None:
    ax.set_xlim(xlim)
  if ylim is not None:
    ax.set_ylim(ylim)

  if invy:
    ax.yaxis.tick_right()
    
  ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.{}f'.format(d)))
  ax.grid(True)

"""\
Function to calculate the cut points P1 and P2, with respect to a circle with center at 'c' 
and radius 'r', which generate the two tangent lines passing through P1 -
"""
def cone_params(c, r, p0):
  cx, cy = c
  x0, y0 = np.array(p0) - np.array(c)

  d0 = np.sqrt(x0**2 + y0**2)
  if d0 > r:
    p1 = r**2/d0**2 * np.array([x0,y0]) + r/d0**2 * np.sqrt(d0**2 - r**2) * np.array([-y0,x0]) + np.array([cx,cy])
    p2 = r**2/d0**2 * np.array([x0,y0]) - r/d0**2 * np.sqrt(d0**2 - r**2) * np.array([-y0,x0]) + np.array([cx,cy])
    return p1, p2
  else:
    return None, None
  
"""\
Create a new directory if it doesn't exist -
"""
def createDir(dir):
  try:
    os.mkdir(dir)
    print("Directory '{}' created!".format(dir))
  except:
    print("The directory '{}' already exists!".format(dir))