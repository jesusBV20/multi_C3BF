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
E = np.array([[0, 1],[-1, 0]])

COLOR_RBT = color_palette()[0]
COLOR_OBS = color_palette()[3]

LW, LH = 12, 4.8
FIGSIZE = [LW, LH]

PX_LIMS = [-70,70]
PY_LIMS = [-50,50]
P_LIMS = [-70,70]

WDATA_LIMS = [-4,4]
PDATA_LIMS = [0,10]
LGH_LIMS = [-500,500]

"""\
###########
"""
class sim_3:
  def __init__(self, n_agents=12, n_obs=6, tf=100, dt=1/60, a=20, b=12, area=140**2,
                     s=1, ke=0.1, kn=2, r=0.5, gamma=1, d=0.87, d_obs=0.89, t_cbf=0):
    self.dt = dt * 1000  # in miliseconds
    self.tf = tf * 1000  # in miliseconds
    self.data = {"pf": None, "phif": None, "prelnorm": None, "omega": None, "lgh": None,
                 "prelvi": None, "vjevi" : None}

    # Trayectory parameters and generation
    self.s = s
    XYoff, alpha = [0, 0], 0
    
    self.gvf_traj = gvf_ellipse(XYoff,alpha,a,b)
    self.area = area

    # CBF parameters
    self.obs_list = np.arange(0,n_obs)
    self.r = r
    
    rho = lambda prel, k: (np.sqrt(prel.T@prel)**(d)/r**(d-1))
    rho_dot = lambda prel, vrel, k: (d * np.sqrt(prel.T@prel)**(d-1)/r**(d - 1) \
                                       * prel.T@vrel/np.sqrt(prel.T@prel))
    
    rho_obs = lambda prel, k: (np.sqrt(prel.T@prel)**(d_obs)/r**(d-1))
    rho_dot_obs = lambda prel, vrel, k: (d_obs * np.sqrt(prel.T@prel)**(d_obs-1)/r**(d_obs - 1) \
                                               * prel.T@vrel/np.sqrt(prel.T@prel))

    # -- Initial state --

    # Initial states of the robots
    p0_rbt = np.array(self.gvf_traj.param_points(pts=n_agents-n_obs+1)).T[:-1,:]*2.1
    p0_obs = np.array(self.gvf_traj.param_points(pts=n_obs+1)).T[:-1,:]*1.2
    p0 = np.vstack([p0_obs,p0_rbt])

    v0 = np.linspace(0,5,n_agents) + 2
    v0 = v0[:,None]
    np.random.shuffle(v0)
    self.v = v0

    p0 = p0 * (v0/np.max(v0)/4 + 1) # ensure assumption 4
    p0_obs = p0[:n_obs]
    p0_rbt = p0[n_obs:]

    v0_rbt = np.copy(v0)
    v0_rbt[0:n_obs,:] = v0_rbt[0:n_obs,:]*0
    
    t = self.s*self.gvf_traj.grad_phi(p0_rbt)@E.T
    phi0_rbt = np.arctan2(t[:,1],t[:,0])
    t = -self.s*self.gvf_traj.grad_phi(p0_obs)@E.T
    phi0_obs = np.arctan2(t[:,1],t[:,0])
    phi0 = np.hstack([phi0_obs,phi0_rbt])

    # Initial state vector
    x0 = [p0, v0_rbt, phi0]
    x0_obs = [p0_obs, v0[0:n_obs], phi0_obs]
    
    # -------------------------------------------------
    # Robots simulator init
    self.sim = simulator(self.gvf_traj, n_agents, x0, self.dt, t_cbf=t_cbf, 
                         rho=rho, rho_dot=rho_dot, obs=self.obs_list)
    self.sim.set_params(s, ke, kn, r, gamma)

    # Obstacles simulator init
    self.sim_obs = simulator(self.gvf_traj, n_obs, x0_obs, self.dt, t_cbf=t_cbf, 
                      rho=rho_obs, rho_dot=rho_dot_obs)
    self.sim_obs.set_params(-s, ke, kn, r, gamma)

    # -------------------------------------------------
    # Generating vector field
    self.gvf_traj.vector_field(XYoff, area, s, ke)

    # Title of the plots
    self.title = r"$N$ = {0:d}, $r$ = {1:.1f}, $\kappa$ = $h^3$".format(self.sim.N, self.r)
    self.title = self.title + r", $k_e$ = {0:.1f}, $k_n$ = {1:.1f}".format(ke, kn)
    self.title = self.title + r", $d_b$ = {0:.2f}, $d_r$ = {1:.2f}".format(d, d_obs)


  """\
  Function to launch the numerical simulation
  """
  def numerical_simulation(self):
    its = int(self.tf/self.dt)
    n_obs = self.sim_obs.N
    
    pfdata   = np.empty([its,self.sim.N, 2])
    phidata  = np.empty([its,self.sim.N])
    preldata = np.empty([its,self.sim.N,self.sim.N,2])
    omega    = np.empty([its,self.sim.N])
    lgh    = np.empty([its,self.sim.N,self.sim.N])
    prelvi   = np.empty([its,self.sim.N,self.sim.N])
    vjevi    = np.empty([its,self.sim.N,self.sim.N])

    for i in tqdm(range(its)):
      # Obstacle sim data
      pfdata[i,0:n_obs,:] = self.sim_obs.pf
      phidata[i,0:n_obs]  = self.sim_obs.phif
      omega[i,0:n_obs]    = self.sim_obs.w

      self.sim.pf[0:n_obs,:] = self.sim_obs.pf
      self.sim.phif[0:n_obs] = self.sim_obs.phif
      self.sim.w[0:n_obs]    = self.sim_obs.w

      # Obstacles simulator euler step integration
      self.sim_obs.int_euler()

      # ---
      # Robots sim data
      pfdata[i,n_obs:,:] = self.sim.pf[n_obs:,:]
      phidata[i,n_obs:]  = self.sim.phif[n_obs:]
      omega[i,n_obs:]    = self.sim.w[n_obs:]
      preldata[i,:,:,:]  = self.sim.p_rel
      lgh[i,:,:] = self.sim.lgh
      prelvi[i,:,:] = self.sim.prelvi
      vjevi[i,:,:] = self.sim.vjevi

      # Robots simulator euler step integration
      self.sim.int_euler()

    # List to numpy arrays for easier indexing
    self.data["pf"] = pfdata
    self.data["phif"] = phidata
    self.data["prelnorm"] = np.linalg.norm(preldata, axis=3)
    self.data["omega"] = omega
    self.data["lgh"] = lgh
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
    prelvi = self.data["prelvi"]
    vjevi = self.data["vjevi"]
    
    # -- Plotting the summary --
    # Figure and grid init
    fig = plt.figure(figsize=FIGSIZE, dpi=dpi)
    grid_outer = plt.GridSpec(1, 2, hspace=0, wspace=0.1)

    grid_main = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = grid_outer[0])
    grid_data = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec = grid_outer[1], wspace = .5)

    main_ax  = fig.add_subplot(grid_main[0,0])
    prel_ax  = fig.add_subplot(grid_data[0, 0], xticklabels=[])
    lgh_ax = fig.add_subplot(grid_data[1, 0], xticklabels=[])
    wdata_ax = fig.add_subplot(grid_data[2, 0])
    prelvi_ax = fig.add_subplot(grid_data[0, 1], xticklabels=[])
    vjevi_ax = fig.add_subplot(grid_data[1, 1], xticklabels=[])
    
    # Axis formatting
    main_ax.set_xlim(PX_LIMS)
    main_ax.set_ylim(PY_LIMS)
    main_ax.set_ylabel(r"$p_y$ (L)")  
    main_ax.set_xlabel(r"$p_x$ (L)")
    main_ax.set_aspect("equal")
    main_ax.grid(True)

    fmt_data_axis(prel_ax,  r"$||p_{ij}||$ [L]", ylim=PDATA_LIMS)
    fmt_data_axis(lgh_ax, r"$L_gh^i(q_{ij})$")
    fmt_data_axis(wdata_ax, r"$\omega [rad/T]$", r"$t$ (T)", ylim=WDATA_LIMS)
    fmt_data_axis(prelvi_ax, r"$\hat p_{ij}^\top E \hat v_i$", ylim=[-1.1,1.1])
    fmt_data_axis(vjevi_ax, r"$\hat v_j^\top E \hat v_i$", r"$t$ (T)", ylim=[-1.1,1.1])

    # -- Main axis plotting
    self.gvf_traj.draw(fig, main_ax)
    main_ax.set_title(self.title, fontdict={'fontsize': 10})

    # Generating unicycle icons
    li = xdata.shape[0] - 1
    for n in range(self.sim.N):

      if n not in self.obs_list:  # Agent icons
        color = COLOR_RBT
        icon_init = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], color, size=1)
        icon_init.set_alpha(0.3)
        icon = unicycle_patch([xdata[li,n], ydata[li,n]], phidata[li,n], color, size=1)

        main_ax.plot(xdata[:,n],ydata[:,n], c=color, ls="-", lw=0.8, zorder=0, alpha=0.2)
        main_ax.add_patch(icon_init)
        main_ax.add_patch(icon)

      else: # Obstacle icons
        color = COLOR_OBS
        icon_init = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], color, size=1)
        icon_init.set_alpha(0.3)
        icon = unicycle_patch([xdata[li,n], ydata[li,n]], phidata[li,n], color, size=1)

        main_ax.plot(xdata[:,n],ydata[:,n], c=color, ls="-", lw=0.8, zorder=0, alpha=0.2)
        main_ax.add_patch(icon_init)
        main_ax.add_patch(icon)

    # -- Data axis plotting
    time_vec = np.linspace(0, self.tf/1000, int(self.tf/self.dt))

    # Zero lines
    prel_ax.axhline(self.r,  c="black", ls="--", lw=1.2, zorder=0, alpha=1)
    wdata_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=0.5)
    prel_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=0.5)

    prel_ax.axhline(0,0,  c="blue", ls="-", lw=1.2, label=r"$||p_{ij}^b||$")
    prel_ax.axhline(0,0,  c="red", ls="-", lw=1.2, label=r"$||p_{ij}^r||$")
    prel_ax.axhline(0,0,  c="black", ls="-", lw=1.2, label=r"$||p_i^b - p_j^r||$")

    # Plotting data
    for n in range(self.sim.N):
      if n in self.obs_list:
        wdata_ax.plot(time_vec, omega[:,n], c=COLOR_OBS, lw=1.2, alpha=0.2)
      else:
        wdata_ax.plot(time_vec, omega[:,n], c=COLOR_RBT, lw=1.2, alpha=0.2)

      for k in range(self.sim.N):
        if k > n:
          if n in self.obs_list:
            if k in self.obs_list:
              prel_ax.plot(time_vec, preldata[:,k,n], c=COLOR_OBS, lw=1.2, alpha=0.2, zorder=2)
            else:
              prel_ax.plot(time_vec, preldata[:,k,n], c="k", lw=1.2, alpha=0.09, zorder=3)
          
          else:
            prel_ax.plot(time_vec, preldata[:,k,n], c=COLOR_RBT, lw=1.2, alpha=0.2, zorder=2)

      n_obs = self.sim_obs.N
      for k in range(self.sim.N - n_obs):
        if k > n:
          if self.v[n+n_obs] > self.v[k+n_obs]:
            lgh_ax.plot(time_vec, lgh[:,n_obs+k,n_obs+n], c=COLOR_RBT, lw=1.2, alpha=0.2, zorder=2)
            prelvi_ax.plot(time_vec, prelvi[:,k,n], c=COLOR_RBT, lw=1.2, alpha=0.05, zorder=2)
            vjevi_ax.plot(time_vec, vjevi[:,k,n], c=COLOR_RBT, lw=1.2, alpha=0.05, zorder=2)
    
    prel_ax.legend(loc="upper left", ncol=3, fancybox=True, framealpha=1, fontsize=6)
    
    # Save the figure
    plt.savefig(os.path.join(output_folder, "plot__{0}_{1}_{2}__{3}_{4}_{5}.png".format(*time.localtime()[0:6])))


  """\
  Animation function update
  """
  def animate(self, i, xdata, ydata, phidata, n_tail):
    for n in range(self.sim.N):
      self.icons_plt[n].remove()
      if n not in self.obs_list:
        self.icons_plt[n] = unicycle_patch([xdata[i,n], ydata[i,n]], phidata[i,n], COLOR_RBT, size=1)
        self.icons_plt[n].set_zorder(3)

      else:
        self.icons_plt[n] = unicycle_patch([xdata[i,n], ydata[i,n]], phidata[i,n], COLOR_OBS, size=1)
        self.icons_plt[n].set_zorder(3)

      self.anim_axis.add_patch(self.icons_plt[n])

      # Dibujamos la cola
      if i > n_tail:
        self.lines_plt[n].set_data(xdata[i-n_tail:i,n], ydata[i-n_tail:i,n])
      else:
        self.lines_plt[n].set_data(xdata[0:i,n], ydata[0:i,n])

    self.pline.set_xdata(i*self.dt/1000)
    self.lline.set_xdata(i*self.dt/1000)
    self.wline.set_xdata(i*self.dt/1000)

  
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
    self.anim_axis.set_xlim(P_LIMS)
    self.anim_axis.set_ylim(P_LIMS)
    self.anim_axis.set_ylabel(r"$p_y$ (L)")  
    self.anim_axis.set_xlabel(r"$p_x$ (L)")
    self.anim_axis.grid(True)

    fmt_data_axis(prel_ax, ylabel = r"$||p_{ij}||$ [L]", ylim=PDATA_LIMS)
    fmt_data_axis(lgh_ax, ylabel = r"$L_g h^i(q_{ij})$")
    fmt_data_axis(wdata_ax, r"$\omega [rad/T]$", r"$t$ (T)", ylim=WDATA_LIMS)

    # -- Main axis plotting
    self.gvf_traj.draw(fig, self.anim_axis)

    self.lines_plt = []
    self.icons_plt = []
    self.icons_col_plt = []

    for n in range(self.sim.N):
      icon = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], COLOR_RBT)

    # Draw unicycle icons
    for n in range(self.sim.N):
      
      if n not in self.obs_list: # Agent icons
        icon = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], COLOR_RBT, size=1)
        line, = self.anim_axis.plot(xdata[:,n], ydata[:,n], c=COLOR_RBT, ls="-", lw=0.8)

        self.anim_axis.add_patch(icon)
        self.lines_plt.append(line)
        self.icons_plt.append(icon)

      else: # Obstacle icons
        icon = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], COLOR_OBS, size=1)
        line, = self.anim_axis.plot(xdata[:,n], ydata[:,n], c=COLOR_OBS, ls="-", lw=0.8)

        self.anim_axis.add_patch(icon)
        self.lines_plt.append(line)
        self.icons_plt.append(icon)

    txt_title = self.anim_axis.set_title(self.title)

    # -- Main axis plotting
    time_vec = np.linspace(0, self.tf/1000, int(self.tf/self.dt))

    # Zero lines
    prel_ax.axhline(self.r,  c="black", ls="--", lw=1.2, zorder=0, alpha=1)
    wdata_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=0.5)

    prel_ax.axhline(0,0,  c="blue", ls="-", lw=1.2, label=r"$||p_{ij}^b||$")
    prel_ax.axhline(0,0,  c="red", ls="-", lw=1.2, label=r"$||p_{ij}^r||$")
    prel_ax.axhline(0,0,  c="black", ls="-", lw=1.2, label=r"$||p_i^b - p_j^r||$")

    # Plotting data
    for n in range(self.sim.N):
      if n in self.obs_list:
        wdata_ax.plot(time_vec, omega[:,n], c=COLOR_OBS, lw=1.2, alpha=0.2)
      else:
        wdata_ax.plot(time_vec, omega[:,n], c=COLOR_RBT, lw=1.2, alpha=0.2)

      for k in range(self.sim.N):
        if k > n:
          if n in self.obs_list:
            if k in self.obs_list:
              prel_ax.plot(time_vec, preldata[:,k,n], c=COLOR_OBS, lw=1.2, alpha=0.2, zorder=2)
              if self.v[n] > self.v[k]:
                lgh_ax.plot(time_vec, lgh[:,k,n], c=COLOR_OBS, lw=1.2, alpha=0.1, zorder=2)
            else:
              prel_ax.plot(time_vec, preldata[:,k,n], c="k", lw=1.2, alpha=0.09, zorder=3)
          
          else:
            prel_ax.plot(time_vec, preldata[:,k,n], c=COLOR_RBT, lw=1.2, alpha=0.2, zorder=2)
            if self.v[n] > self.v[k]:
              lgh_ax.plot(time_vec, lgh[:,k,n], c=COLOR_RBT, lw=1.2, alpha=0.1, zorder=2)

    self.pline = prel_ax.axvline(0, c="black", ls="--", lw=1.2)
    self.lline = lgh_ax.axvline(0, c="black", ls="--", lw=1.2)
    self.wline = wdata_ax.axvline(0, c="black", ls="--", lw=1.2)

    prel_ax.legend(loc="upper left", ncol=3, fancybox=True, framealpha=1, fontsize=10)

    # -- Animation --
    # Init of the animation class
    anim = FuncAnimation(fig, self.animate, fargs=(xdata, ydata, phidata, n_tail),
                         frames=tqdm(range(frames), initial=1, position=0), interval=1/fps*1000)

    # Generate and save the animation
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Jesús Bautista Villar'), bitrate=10000)
    anim.save(os.path.join(output_folder, "anim__{0}_{1}_{2}__{3}_{4}_{5}.mp4".format(*time.localtime()[0:6])), 
              writer = writer)