#!/usr/bin/env python #
"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import os
import time
import warnings
from tqdm import tqdm

import numpy as np
from numpy import linalg as LA

# Graphic tools
import matplotlib.pyplot as plt
from seaborn import color_palette
import matplotlib.patches as patches

# Animation tools
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML


# CBF_tools
from .utils.toolbox import unicycle_patch, fmt_data_axis
from .utils.simulator import simulator

# GVF trajectories
from .gvf_traj.gvf_traj_ellipse  import gvf_ellipse

# -- Global variables --
E = np.array([[0, 1],[-1, 0]])

COLOR_RBT = color_palette()[0]

LW, LH = 12, 12
FIGSIZE = [LW, LH]

PX_LIMS = [-35,35]
PY_LIMS = [-35,35]
P_LIMS = [-35,35]

WDATA_LIMS = [-4,4]
PDATA_LIMS = [0,10]


"""\
###########
"""
class sim_4:
  def __init__(self, n_agents=12, tf=100, dt=1/60, a=18, b=7, area=70**2,
                     s=1, ke=0.1, kn=2, r=0.5, gamma=1, d=0.83, t_cbf=5):
    self.dt = dt * 1000 / 2 # in miliseconds
    self.tf = tf * 1000  # in miliseconds
    self.data = {"pf": None, "phif": None, "prelnorm": None, "omega": None, "lgh": None}

    # Trayectory parameters and generation
    self.s = s
    XYoff, alpha = [0, 0], 0
    
    self.gvf_traj = gvf_ellipse(XYoff,alpha,a,b)
    self.area = area

    # CBF parameters
    self.r = r
    
    rho = lambda prel, k: (np.sqrt(prel.T@prel)**(d)/r**(d-1))
    rho_dot = lambda prel, vrel, k: (d * np.sqrt(prel.T@prel)**(d-1)/r**(d - 1) \
                                       * prel.T@vrel/np.sqrt(prel.T@prel))


    # -- Initial state --

    # Initial states of the robots
    p0 = np.array(self.gvf_traj.param_points(pts=n_agents+1)).T[:-1,:]*0.3

    v0 = np.linspace(0,5,n_agents) + 2
    # if n_agents>3:
    #   v0[-3:] = v0[-3:] + 15
    v0 = v0[:,None]
    np.random.shuffle(v0)
    self.v = v0

    p0 = p0 * (v0/np.max(v0)*1.2 + 1) # ensure lemma 1 conditions

    t = self.s*self.gvf_traj.grad_phi(p0)@E.T
    phi0 = np.arctan2(t[:,1],t[:,0])

    # Initial state vector
    x0 = [p0, v0, phi0]
    
    # -------------------------------------------------
    # Robots simulator init
    self.sim = simulator(self.gvf_traj, n_agents, x0, self.dt, t_cbf=t_cbf, 
                         rho=rho, rho_dot=rho_dot)
    self.sim.set_params(s, ke, kn, r, gamma)

    # -------------------------------------------------
    # Generating vector field
    self.gvf_traj.vector_field(XYoff, area, s, ke)

    # Title of the plots
    self.title = r"$N$ = {0:d}, $r$ = {1:.1f}, $\kappa$ = {2:.2f} $h^3$".format(self.sim.N, self.r, gamma)
    self.title = self.title + r", $k_e$ = {0:.1f}, $k_n$ = {1:.1f}".format(ke, kn)
    self.title = self.title + r", $d$ = {0:.2f}".format(d)


  """\
  Function to launch the numerical simulation
  """
  def numerical_simulation(self):
    its = int(self.tf/self.dt)

    pfdata   = np.empty([its,self.sim.N, 2])
    phidata  = np.empty([its,self.sim.N])

    for i in tqdm(range(its)):

      # Robots sim data
      pfdata[i,:,:] = self.sim.pf
      phidata[i,:]  = self.sim.phif

      # Robots simulator euler step integration
      self.sim.int_euler()

    # List to numpy arrays for easier indexing
    self.data["pf"] = pfdata
    self.data["phif"] = phidata


  """\
  Function to generate the summary graphical plot of the whole simulation
  """
  def plot_summary(self, output_folder, dpi=200):
    # -- Extract data fields from data dictonary --
    xdata    = self.data["pf"][:,:,0]
    ydata    = self.data["pf"][:,:,1]
    phidata  = self.data["phif"]
    
    # -- Plotting the summary --
    # Figure and grid init
    fig = plt.figure(figsize=FIGSIZE, dpi=dpi)
    grid = plt.GridSpec(1, 1, hspace=0.1, wspace=0.4)

    main_ax  = fig.add_subplot(grid[:, :])

    # Axis formatting
    main_ax.set_xlim(PX_LIMS)
    main_ax.set_ylim(PY_LIMS)
    main_ax.grid(True)
    main_ax.set_facecolor('black')

    # Generating unicycle icons
    li = xdata.shape[0] - 1
    for n in range(self.sim.N):

      color = COLOR_RBT
      icon_init = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], color, size=1)
      icon_init.set_alpha(0.3)
      icon = unicycle_patch([xdata[li,n], ydata[li,n]], phidata[li,n], color, size=1)

      main_ax.plot(xdata[:,n],ydata[:,n], c=color, ls="-", lw=0.8, zorder=0, alpha=0.2)
      main_ax.add_patch(icon_init)
      main_ax.add_patch(icon)

    # Save the figure
    plt.savefig(os.path.join(output_folder, "plot__{0}_{1}_{2}__{3}_{4}_{5}.png".format(*time.localtime()[0:6])))


  """\
  Animation function update
  """
  def animate(self, i, xdata, ydata, phidata, n_tail):
    for n in range(self.sim.N):
      self.icons_plt[n].remove()
      self.icons_plt[n] = unicycle_patch([xdata[i,n], ydata[i,n]], phidata[i,n], COLOR_RBT, size=1)
      self.icons_plt[n].set_alpha(self.icon_alphas[n])
      self.icons_plt[n].set_zorder(3)

      self.anim_axis.add_patch(self.icons_plt[n])

      # Dibujamos la cola
      dn = 2
      if i > dn:
        i_ = i - dn
        if i > n_tail + dn:
          self.lines_plt[n].set_data(xdata[i_-n_tail:i_,n], ydata[i_-n_tail:i_,n])
        else:
          self.lines_plt[n].set_data(xdata[0:i_,n], ydata[0:i_,n])

  
  """\
  Funtion to generate the full animation of the simulation
  """
  def generate_animation(self, output_folder, tf_anim=None, res=1920, n_tail=1000):
    if tf_anim is None:
      tf_anim = self.tf
    
    fps = 1000/self.dt
    frames = int(tf_anim/self.dt-1)

    print("Animation parameters: ", {"fps":fps, "tf":tf_anim/1000, "frames":frames})

    #warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

    # -- Extract data fields from data dictonary --
    xdata    = self.data["pf"][:,:,0]
    ydata    = self.data["pf"][:,:,1]
    phidata  = self.data["phif"]
    
    # -- Generating the animation --
    # Figure and grid init
    fig = plt.figure(figsize=FIGSIZE, dpi=res/FIGSIZE[0])
    grid = plt.GridSpec(1, 1, hspace=0.1, wspace=0.4)

    self.anim_axis  = fig.add_subplot(grid[:, :])

    # Axis formatting
    #fig.set_facecolor('black')
    self.anim_axis.set_xlim(P_LIMS)
    self.anim_axis.set_ylim(P_LIMS)
    #self.anim_axis.set_facecolor('black')
    self.anim_axis.set_xticks([])
    self.anim_axis.set_yticks([])

    # Main axis plotting
    self.lines_plt = []
    self.icons_plt = []
    self.icons_col_plt = []

    for n in range(self.sim.N):
      icon = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], COLOR_RBT)

    # Draw unicycle icons
    self.icon_alphas = 0.4 + (np.random.rand(self.sim.N)+1)/4
    for n in range(self.sim.N):
      
      # Agent icons
      icon = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], COLOR_RBT, size=1)
      icon.set_alpha(self.icon_alphas[n])
      line, = self.anim_axis.plot(0, 0, c=COLOR_RBT, ls="-", lw=0.8, alpha=0.3)

      self.anim_axis.add_patch(icon)
      self.lines_plt.append(line)
      self.icons_plt.append(icon)

    # -- Animation --
    # Init of the animation class
    anim = FuncAnimation(fig, self.animate, fargs=(xdata, ydata, phidata, n_tail),
                         frames=tqdm(range(frames), initial=1, position=0), interval=1/fps*1000)

    # Generate and save the animation
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Jesús Bautista Villar'), bitrate=10000)
    anim.save(os.path.join(output_folder, "anim__{0}_{1}_{2}__{3}_{4}_{5}.mp4".format(*time.localtime()[0:6])), 
              writer = writer)