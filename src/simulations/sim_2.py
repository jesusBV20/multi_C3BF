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
import matplotlib.gridspec as gridspec

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
# -90º rotation matrix
E = np.array([[0, 1],[-1, 0]])
# +90º rotation matrix
E_pos = np.array([[0, -1],[1, 0]]) 

COLOR_RBT = color_palette()[0]
COLOR_OBS = color_palette()[3]

LW, LH = 12, 4.8
FIGSIZE = [LW, LH]

PX_LIMS = [-100,50]
PY_LIMS = [-65,65]

PX_LIMS_A = [-100,50]
PY_LIMS_A = [-80,70]

WDATA_LIMS = [-3,3]
PDATA_LIMS = [0,5]
LGH_LIMS = [-500,500]

"""\
###########
"""
class sim_2:
  def __init__(self, n_agents=12, tf=100, dt=1/60, area=210**2,
                     s=1, ke=0.3, kn=2, r=0.5, gamma=1, d=0.6, t_cbf=0):
    self.dt = dt * 1000  # in miliseconds
    self.tf = tf * 1000  # in miliseconds
    self.data = {"pf": None, "phif": None, "prelnorm": None, "omega": None, 
                 "lgh": None, "pfship": None, "omegasafe": None, 
                 "prelvi": None, "vjevi" : None}

    # Trayectory parameters and generation
    self.s = s
    XYoff = [0, 0]
    
    self.gvf_traj = gvf_ellipse(XYoff,np.pi/4,25,13)
    #self.gvf_traj = gvf_ellipse(XYoff,0,40,25)
    #self.gvf_traj = gvf_eight(XYoff,r1,r2)
    self.area = area

    # CBF parameters
    self.r = r
    
    rho = lambda prel, k: (np.sqrt(prel.T@prel)**(d)/r**(d-1))
    rho_dot = lambda prel, vrel, k: (d * np.sqrt(prel.T@prel)**(d-1)/r**(d - 1) \
                                       * prel.T@vrel/np.sqrt(prel.T@prel))

    # -- Initial state --
    # Initial state of the battleship
    self.r_ship = 4
    self.p_ship = np.array([-80,-10])
    self.v_ship = 2
    self.phi_ship = np.pi/2
    
    # Initial states of the robots
    p0 = np.ones((n_agents,2)) * self.p_ship

    v0 = np.linspace(0,5,n_agents) + 6
    v0 = v0[:,None]
    np.random.shuffle(v0)
    v0[-3:] = v0[-3:] + 10
    self.v = np.copy(v0)
    phi0 = np.ones(n_agents) * np.pi/2

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
    self.title = r"$N$ = {0:d}, $r$ = {1:.1f}".format(self.sim.N, self.r)
    if gamma == 1:
      self.title = self.title + r", $\kappa$ = $h^3$"
    else:
      self.title = self.title + r", $\kappa$ = {0:.2f} $h^3$".format(gamma)
    self.title = self.title + r", $k_e$ = {0:.1f}, $k_n$ = {1:.1f}".format(ke, kn)
    self.title = self.title + r", $d$ = {0:.2f}".format(d)

  """\
  Function to launch the numerical simulation
  """
  def numerical_simulation(self):
    its = int(self.tf/self.dt)
    
    pf_ship = np.copy(self.p_ship)
    vf_ship = self.v_ship * np.array([np.cos(self.phi_ship), np.sin(self.phi_ship)])

    pfshipdata = np.empty([its, 2])
    pfdata   = np.empty([its,self.sim.N, 2])
    phidata  = np.empty([its,self.sim.N])
    preldata = np.empty([its,self.sim.N,self.sim.N])
    vreldata = np.empty([its,self.sim.N,self.sim.N,2])
    omega    = np.empty([its,self.sim.N])
    omegasf  = np.empty([its,self.sim.N])
    lgh      = np.empty([its,self.sim.N,self.sim.N])
    prelvi   = np.empty([its,self.sim.N,self.sim.N])
    vjevi    = np.empty([its,self.sim.N,self.sim.N])

    n_check = 1
    self.sim.vf[n_check:] = np.zeros((self.sim.N-n_check,1))
    for i in tqdm(range(its)):
      # Robots sim data
      pfdata[i,:,:] = self.sim.pf
      phidata[i,:]  = self.sim.phif
      omega[i,:]    = self.sim.w
      preldata[i,:,:]  = np.linalg.norm(self.sim.p_rel, axis=2)
      vreldata[i,:,:,:]  = self.sim.v_rel
      omegasf[i,:] = self.sim.omega_safe
      prelvi[i,:,:] = self.sim.prelvi
      vjevi[i,:,:] = self.sim.vjevi

      if i*self.dt > 20:
        lgh[i,:,:] = self.sim.lgh

      pfshipdata[i,:] = pf_ship

      # Activate agents one by one
      if n_check < self.sim.N:
        if (preldata[i,0:n_check,n_check] > self.r*1.5).all():
          self.sim.vf[n_check] = self.v[n_check]
          n_check = n_check + 1
      
      # Move the ship
      pf_ship = pf_ship + vf_ship * self.dt/1000
      self.sim.pf[n_check:,:] = np.ones((self.sim.N-n_check,2)) * pf_ship

      # Robots simulator euler step integration
      self.sim.int_euler()

    # List to numpy arrays for easier indexing
    self.data["pf"] = pfdata
    self.data["phif"] = phidata
    self.data["prelnorm"] = preldata
    self.data["omega"] = omega
    self.data["lgh"] = lgh
    self.data["pfship"] = pfshipdata
    self.data["omegasafe"] = omegasf
    self.data["prelvi"] = prelvi
    self.data["vjevi"] = vjevi

  """\
  Function to generate the summary graphical plot of the whole simulation
  """
  def plot_summary(self, output_folder, dpi=200):
    # -- Extract data fields from data dictonary --
    xdata    = self.data["pf"][:,:,0]
    ydata    = self.data["pf"][:,:,1]
    phidata  = self.data["phif"]
    preldata = self.data["prelnorm"]
    omega    = self.data["omega"]
    lgh    = self.data["lgh"]
    pfshipdata = self.data["pfship"]
    prelvi = self.data["prelvi"]
    vjevi = self.data["vjevi"]

    # -- Plotting the summary --
    # Figure and grid init
    fig = plt.figure(figsize=FIGSIZE, dpi=dpi)
    grid_outer = plt.GridSpec(1, 2, hspace=0, wspace=0.1)

    grid_main = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = grid_outer[0])
    grid_data = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec = grid_outer[1], wspace = .5)

    main_ax  = fig.add_subplot(grid_main[0,0])
    prel_ax  = fig.add_subplot(grid_data[0, 0], xticklabels=[])
    lgh_ax = fig.add_subplot(grid_data[1, 0])
    wdata_ax = fig.add_subplot(grid_data[0, 1], xticklabels=[])
    prelvi_ax = fig.add_subplot(grid_data[1, 1])
    # vjevi_ax = fig.add_subplot(grid_data[1, 1], xticklabels=[])
    
    # Axis formatting
    main_ax.set_xlim(PX_LIMS)
    main_ax.set_ylim(PY_LIMS)
    main_ax.set_ylabel(r"$p_y$ [L]")  
    main_ax.set_xlabel(r"$p_x$ [L]")
    main_ax.set_aspect("equal")
    main_ax.grid(True)

    fmt_data_axis(prel_ax,  r"$||p_{ij}||$ [L]", title="a)", ylim=PDATA_LIMS)
    fmt_data_axis(lgh_ax, r"$L_gh^i(q_{ij})$", r"$t$ [T]", title="b)")
    fmt_data_axis(wdata_ax, r"$\omega$ [rad/T]", title="c)", ylim=WDATA_LIMS)
    fmt_data_axis(prelvi_ax, r"$\hat p_{ij}^\top E \hat v_i$", title="d)", ylim=[-1.1,1.1])
    #fmt_data_axis(vjevi_ax, r"$\hat v_j^\top E \hat v_i$", r"$t$ (T)", ylim=[-1.1,1.1])

    # -- Main axis plotting
    self.gvf_traj.draw(fig, main_ax)
    main_ax.set_title(self.title, fontdict={'fontsize': 10})

    # Generating unicycle icons
    li = xdata.shape[0] - 1
    for n in range(self.sim.N):

      # Agent icons
      color = COLOR_RBT
      icon_init = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], color, size=1)
      icon_init.set_alpha(0.3)
      icon = unicycle_patch([xdata[li,n], ydata[li,n]], phidata[li,n], color, size=1)

      main_ax.plot(xdata[:,n],ydata[:,n], c=color, ls="-", lw=0.8, zorder=0, alpha=0.1)
      main_ax.add_patch(icon_init)
      main_ax.add_patch(icon)

    # Ship icon
    icon_ship_init = patches.Circle([pfshipdata[0,0], pfshipdata[0,1]], self.r_ship, alpha=0.2, fill=True,
                                      color="gray", ls="-", lw=0)
    icon_ship = patches.Circle([pfshipdata[li,0], pfshipdata[li,1]], self.r_ship, alpha=0.7, fill=True,
                                color="gray", ls="-", lw=0)
    main_ax.add_patch(icon_ship_init)
    main_ax.add_patch(icon_ship) 

    # -- Data axis plotting
    time_vec = np.linspace(0, self.tf/1000, int(self.tf/self.dt))

    # Zero lines
    prel_ax.axhline(self.r,  c="black", ls="--", lw=1.2, zorder=0, alpha=1)
    wdata_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=0.5)
    lgh_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=1)
    prelvi_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=1)
    #vjevi_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=1)

    # Plotting data
    for n in range(self.sim.N):
      wdata_ax.plot(time_vec, omega[:,n], c=COLOR_RBT, lw=1.2, alpha=0.2)

      for k in range(self.sim.N):
        if k > n:
          prel_ax.plot(time_vec, preldata[:,k,n], c=COLOR_RBT, lw=1.2, alpha=0.2, zorder=2)
          if self.v[n] > self.v[k]:
            lgh_ax.plot(time_vec, lgh[:,k,n], c=COLOR_RBT, lw=1.2, alpha=0.1, zorder=2)
            prelvi_ax.plot(time_vec, prelvi[:,k,n], c=COLOR_RBT, lw=1.2, alpha=0.05, zorder=2)
            # vjevi_ax.plot(time_vec, vjevi[:,k,n], c=COLOR_RBT, lw=1.2, alpha=0.05, zorder=2)

    # Save the figure
    plt.savefig(os.path.join(output_folder, "plot__{0}_{1}_{2}__{3}_{4}_{5}.png".format(*time.localtime()[0:6])))


  """\
  Animation function update
  """
  def animate(self, i, xdata, ydata, phidata, p_ship, n_tail):
    for n in range(self.sim.N):
      # Move the agents
      self.icons_plt[n].remove()
      self.icons_plt[n] = unicycle_patch([xdata[i,n], ydata[i,n]], phidata[i,n], COLOR_RBT, size=1)
      self.icons_plt[n].set_zorder(3)
      self.anim_axis.add_patch(self.icons_plt[n])

      # Move the ship
      self.icon_ship.set(center = [p_ship[i,0], p_ship[i,1]])

      # Draw the tail
      if i > n_tail:
        self.lines_plt[n].set_data(xdata[i-n_tail:i,n], ydata[i-n_tail:i,n])
      else:
        self.lines_plt[n].set_data(xdata[0:i,n], ydata[0:i,n])

    # Move the data axis time lines
    self.pline.set_xdata(i*self.dt/1000)
    self.wline.set_xdata(i*self.dt/1000)
    self.lghline.set_xdata(i*self.dt/1000)

  
  """\
  Funtion to generate the full animation of the simulation
  """
  def generate_animation(self, output_folder, tf_anim=None, res=1920, n_tail=50):
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
    preldata = self.data["prelnorm"]
    omega    = self.data["omega"]
    lgh    = self.data["lgh"]
    pfshipdata = self.data["pfship"]

    # -- Generating the animation --
    # Figure and grid init
    figsize = (16,9)

    fig = plt.figure(figsize=figsize, dpi=res/figsize[0])
    grid = plt.GridSpec(3, 5, hspace=0.1, wspace=0.4)

    self.anim_axis  = fig.add_subplot(grid[:, 0:3])
    prel_ax  = fig.add_subplot(grid[0, 3:5], xticklabels=[])
    lgh_ax = fig.add_subplot(grid[1, 3:5], xticklabels=[])
    wdata_ax = fig.add_subplot(grid[2, 3:5])

    # Axis formatting
    self.anim_axis.set_xlim(PX_LIMS_A)
    self.anim_axis.set_ylim(PY_LIMS_A)
    self.anim_axis.set_ylabel(r"$p_y$ (L)")  
    self.anim_axis.set_xlabel(r"$p_x$ (L)")
    self.anim_axis.grid(True)

    fmt_data_axis(prel_ax, ylabel = r"$||p_{ij}||$ [L]", ylim=PDATA_LIMS)
    fmt_data_axis(lgh_ax, r"$L_gh^i$", r"$t$ (T)")
    fmt_data_axis(wdata_ax, r"$\omega [rad/T]$", ylim=WDATA_LIMS)

    # -- Main axis plotting
    self.gvf_traj.draw(fig, self.anim_axis)
    self.anim_axis.set_title(self.title)

    self.lines_plt = []
    self.icons_plt = []
    self.icons_col_plt = []

    for n in range(self.sim.N):
      icon = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], COLOR_RBT)

    # Draw unicycle icons
    for n in range(self.sim.N):
      
      # Agent icons
      icon = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], COLOR_RBT, size=1)
      line, = self.anim_axis.plot(xdata[:,n], ydata[:,n], c=COLOR_RBT, ls="-", lw=0.8)

      self.anim_axis.add_patch(icon)
      self.lines_plt.append(line)
      self.icons_plt.append(icon)

    # Ship icon
    self.icon_ship = patches.Circle([pfshipdata[0,0], pfshipdata[0,1]], self.r_ship, alpha=0.7, fill=True,
                                     color="gray", ls="-", lw=0)
    self.anim_axis.add_patch(self.icon_ship) 

    # -- Main axis plotting
    time_vec = np.linspace(0, self.tf/1000, int(self.tf/self.dt))

    # Zero lines
    prel_ax.axhline(self.r,  c="black", ls="--", lw=1.2, zorder=0, alpha=1)
    lgh_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=1)
    wdata_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=1)

    # Plotting data
    for n in range(self.sim.N):
      wdata_ax.plot(time_vec, omega[:,n], c=COLOR_RBT, lw=1.2, alpha=0.2)

      for k in range(self.sim.N):
        if k > n:
          prel_ax.plot(time_vec, preldata[:,k,n], c=COLOR_RBT, lw=1.2, alpha=0.2, zorder=2)
          if self.v[n] > self.v[k]:
            lgh_ax.plot(time_vec, lgh[:,k,n], c=COLOR_RBT, lw=1.2, alpha=0.1, zorder=2)

    self.pline   = prel_ax.axvline(0, c="black", ls="--", lw=1.2)
    self.lghline = lgh_ax.axvline(0, c="black", ls="--", lw=1.2)
    self.wline   = wdata_ax.axvline(0, c="black", ls="--", lw=1.2)

    # -- Animation --
    # Init of the animation class
    anim = FuncAnimation(fig, self.animate, fargs=(xdata, ydata, phidata, pfshipdata, n_tail),
                         frames=tqdm(range(frames), initial=1, position=0), interval=1/fps*1000)

    # Generate and save the animation
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Jesús Bautista Villar'), bitrate=10000)
    anim.save(os.path.join(output_folder, "anim__{0}_{1}_{2}__{3}_{4}_{5}.mp4".format(*time.localtime()[0:6])), 
              writer = writer)