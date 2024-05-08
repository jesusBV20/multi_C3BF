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
from .utils.toolbox import unicycle_patch, fmt_data_axis, cone_params
from .utils.simulator import simulator

# GVF trajectories
from .gvf_traj.gvf_traj_ellipse  import gvf_ellipse

# -- Global variables --
E = np.array([[0, 1],[-1, 0]])

COLOR_RBT = color_palette()[0]

PX_LIMS = [-25.7,25.7]
PY_LIMS = [-25,25]

WDATA_LIMS = [-1.5,1.5]
LDATA_LIMS = [-5,200]

ARR_KW = {"width":0.003, "scale":51, "zorder":2, "alpha":0.9}
COLOR_PLT = [color_palette()[0], "yellowgreen"]
COLOR_LINE_PLT = [color_palette()[0], "darkgreen"]

"""\
###########
"""
class sim_0:
  def __init__(self, n_agents=12, tf=100, dt=1/60, a=15, b=10, area=50**2,
                     s=1, ke=1, kn=2, r=2, gamma=1, d=0.83, t_cbf=0):
    self.dt = dt * 1000  # in miliseconds
    self.tf = tf * 1000  # in miliseconds
    self.data = {"pf": None, "phif": None, "prelnorm": None, 
                 "omega": None, "lgh": None, "rho_val": None, "prelvi": None}

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
    
    # alfa = 1
    # mu = 5
    # sigm = lambda x: 1/(1 + np.exp(alfa*(x - mu)))
    # rho = lambda prel, k: (np.sqrt(prel.T@prel))**(d)/r**(d-1) * sigm(np.sqrt(prel.T@prel))
    # rho_dot = lambda prel, vrel, k: ( (d * np.sqrt(prel.T@prel)**(d-1)/r**(d - 1) \
    #                                      * sigm(np.sqrt(prel.T@prel))
    #                                      + (np.sqrt(prel.T@prel))**(d)/r**(d-1) \
    #                                      * sigm(np.sqrt(prel.T@prel))*(1-sigm(np.sqrt(prel.T@prel))) \
    #                                      * (-alfa) ) \
    #                                      * prel.T@vrel/np.sqrt(prel.T@prel))
    
    # -- Initial state --

    # Initial states of the robots
    n_agents = 2
    p0 = np.array([[0, -11], [-5, 11]])
    v0 =  np.array([[7, 1]]).T
    phi0 = np.pi - np.array([0, -np.pi])

    # Initial state vector
    x0 = [p0, v0, phi0]
    self.v = v0
    
    # -------------------------------------------------
    # Robots simulator init
    self.sim = simulator(self.gvf_traj, n_agents, x0, self.dt, t_cbf=t_cbf, 
                         rho=rho, rho_dot=rho_dot)
    self.sim.set_params(s, ke, kn, r, gamma)

    # -------------------------------------------------
    # Generating vector field
    self.gvf_traj.vector_field(XYoff, area, s, ke)

    # Title of the plots
    self.title = r"$N$ = {0:d}, $r$ = {1:.1f}, $\kappa$ = {2:.4f} $h^3$".format(self.sim.N, self.r, gamma)
    self.title = self.title + r", $k_e$ = {0:.1f}, $k_n$ = {1:.1f}".format(ke, kn)
    self.title = self.title + r", $d$ = {0:.2f}".format(d)


  """\
  Function to launch the numerical simulation
  """
  def numerical_simulation(self):
    its = int(self.tf/self.dt)

    pfdata   = np.empty([its,self.sim.N, 2])
    phidata  = np.empty([its,self.sim.N])
    preldata = np.empty([its,self.sim.N,self.sim.N,2])
    omega    = np.empty([its,self.sim.N])
    lgh      = np.empty([its,self.sim.N,self.sim.N])
    rho_val  = np.empty([its,self.sim.N,self.sim.N])
    prelvi   = np.empty([its,self.sim.N,self.sim.N])

    for i in tqdm(range(its)):

      # Robots sim data
      pfdata[i,:,:] = self.sim.pf
      phidata[i,:]  = self.sim.phif
      omega[i,:]    = self.sim.w
      preldata[i,:,:,:] = self.sim.p_rel
      lgh [i,:,:]       = self.sim.lgh_all
      rho_val[i,:,:]    = self.sim.rho_val
      prelvi[i,:,:] = self.sim.prelvi

      # Robots simulator euler step integration
      self.sim.int_euler()

    # List to numpy arrays for easier indexing
    self.data["pf"] = pfdata
    self.data["phif"] = phidata
    self.data["prelnorm"] = np.linalg.norm(preldata, axis=3)
    self.data["omega"] = omega
    self.data["lgh"] = lgh
    self.data["rho_val"] = rho_val
    self.data["prelvi"] = prelvi


  """\
  Function to generate the summary graphical plot of the whole simulation
  """
  def plot_summary(self, output_folder, res=1920):
    t_list = np.array([5, 7.5, 10]) * 1000

    # -- Extract data fields from data dictonary --
    xdata    = self.data["pf"][:,:,0]
    ydata    = self.data["pf"][:,:,1]
    phidata  = self.data["phif"]
    preldata = self.data["prelnorm"]
    omega    = self.data["omega"]
    lgh    = self.data["lgh"]
    rho_val    = self.data["rho_val"]
    
    ti, tf = t_list[0], t_list[-1]
    li, lf = int(ti/self.dt), int(tf/self.dt)
    
    # -- Plotting the summary --
    # Figure and grid init
    figsize = (16,9)

    fig = plt.figure(figsize=figsize, dpi=res/figsize[0])
    grid = plt.GridSpec(3, 5, hspace=0.1, wspace=0.4)

    main_ax  = fig.add_subplot(grid[:, 0:3])
    prel_ax  = fig.add_subplot(grid[0, 3:5], xticklabels=[])
    lgh_ax = fig.add_subplot(grid[1, 3:5], xticklabels=[])
    wdata_ax = fig.add_subplot(grid[2, 3:5])

    # Format of the axis
    main_ax.set_xlim(PX_LIMS)
    main_ax.set_ylim(PY_LIMS)
    main_ax.set_title(self.title, fontsize=14)
    main_ax.set_ylabel(r"$p_y$ (L)")  
    main_ax.set_xlabel(r"$p_x$ (L)")
    main_ax.grid(True)

    fmt_data_axis(prel_ax, ylabel = r"$||p_{ij}||$ [L]")
    fmt_data_axis(lgh_ax, r"$L_gh^i$", ylim=LDATA_LIMS)
    fmt_data_axis(wdata_ax, r"$\omega [rad/T]$", r"$t$ (T)", ylim=WDATA_LIMS)

    # -- Main axis
    self.gvf_traj.draw(fig, main_ax, width=0.0024, alpha=0.1, lw=2)

    for t in t_list:
        lt = int(t/self.dt)

        p_obs, p_rbt = [xdata[lt,1], ydata[lt,1]], [xdata[lt,0], ydata[lt,0]]
        p_rel = np.array(p_obs) - np.array(p_rbt)

        # Icon
        for n in range(self.sim.N):
          color = COLOR_PLT[n] # color
          icon = unicycle_patch([xdata[lt,n], ydata[lt,n]], phidata[lt,n], color, size=1.2)
          main_ax.add_patch(icon)

        # GRAY CONES
        p1, p2 = cone_params(p_obs, rho_val[lt,0,1], p_rbt)
        cone = patches.Polygon(np.array([p_rbt, p1, p2]),
                              alpha=0.05, color="gray", zorder=1, lw=0)
        main_ax.add_patch(cone)

        # RED CONE
        p1, p2 = cone_params(p_obs - 2*p_rel, rho_val[lt,0,1], p_rbt)
        cone = patches.Polygon(np.array([p_rbt, p1, p2]),
                              alpha=0.2, color="red", zorder=1, lw=0)
        main_ax.add_patch(cone)
        if t == t_list[0]:
          cone.set_label("Col. cone")

        # Drawing vectors
        v_obs, v_rbt = self.v[1], self.v[0]
        phi_rbt = phidata[lt,0]
        phi_obs = phidata[lt,1]
        v_rel = (v_obs*np.array([np.cos(phi_obs), np.sin(phi_obs)]) - \
                v_rbt*np.array([np.cos(phi_rbt), np.sin(phi_rbt)]) )/4

        main_ax.quiver(p_rbt[0], p_rbt[1], v_rel[0], v_rel[1], **ARR_KW, color="red")
        main_ax.quiver(p_rbt[0], p_rbt[1], p_rel[0], p_rel[1], **ARR_KW, color="k")

        # Virtual collision zone
        icon_col = patches.Circle(p_obs, self.r, alpha=0.4, fill=True,
                                  color=COLOR_PLT[1], ls="-", lw=0, zorder=0)
        main_ax.add_patch(icon_col)

        icon_col_real = patches.Circle(p_obs, rho_val[lt,0,1], alpha=0.2, fill=True,
                                  color="gray", ls="-", lw=0, zorder=-1)
        main_ax.add_patch(icon_col_real)

        if t == t_list[0]:
          icon_col.set_label("Real col. zone")
          icon_col_real.set_label("Virtual col. zone")

    # Trace
    for n in range(self.sim.N):
      color = COLOR_PLT[n] # color
      main_ax.plot(xdata[li:lf,n], ydata[li:lf,n], c=color, ls="-", lw=1.2, zorder=0)

    # Legend settings
    main_ax.legend(loc="upper left", ncol=3, fancybox=True, framealpha=1, fontsize=13)

    # -- Data axis
    time_vec = np.linspace(0, self.tf/1000, int(self.tf/self.dt))
    l0 = 2

    # Zero lines
    prel_ax.axhline(self.r,  c="black", ls="--", lw=1.2, zorder=0, alpha=1)
    lgh_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=0.8)
    wdata_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=0.8)

    prel_ax.axvline(0,  c="black", ls="-", lw=1.2, zorder=0, alpha=1)
    lgh_ax.axvline(0, c="black", ls="-", lw=1.2, zorder=0, alpha=1)
    wdata_ax.axvline(0, c="black", ls="-", lw=1.2, zorder=0, alpha=1)

    # t lines
    for t in t_list:
      prel_ax.axvline(t/1000,  c="black", ls="--", lw=1, zorder=0, alpha=1)
      lgh_ax.axvline(t/1000,  c="black", ls="--", lw=1, zorder=0, alpha=1)
      wdata_ax.axvline(t/1000,  c="black", ls="--", lw=1, zorder=0, alpha=1)

    # Plotting data
    for n in range(self.sim.N):
      wdata_ax.plot(time_vec[l0:], omega[l0:,n], c=COLOR_RBT, lw=1.2, alpha=1)

      for k in range(self.sim.N):
        if k > n:
          prel_ax.plot(time_vec[l0:], preldata[l0:,k,n], c=COLOR_RBT, lw=1.2, alpha=1, zorder=2)
          if self.v[n] > self.v[k]:
            lgh_ax.plot(time_vec[l0:], lgh[l0:,k,n], c=COLOR_RBT, lw=1.2, alpha=1, zorder=2)

    # Save the figure
    plt.savefig(os.path.join(output_folder, "plot__{0}_{1}_{2}__{3}_{4}_{5}.png".format(*time.localtime()[0:6])))


  """""""""""""""""""""""""""\
  Animation function update
  """
  def animate(self, i, xdata, ydata, phidata, rho_val, n_tail):
    for n in range(self.sim.N):
      # Icons
      self.icons_plt[n].remove()
      self.icons_plt[n] = unicycle_patch([xdata[i,n], ydata[i,n]], phidata[i,n], COLOR_PLT[n], size=1)
      self.icons_plt[n].set_zorder(3)
      self.anim_axis.add_patch(self.icons_plt[n])

      # Trace
      if i > n_tail:
        self.lines_plt[n].set_data(xdata[i-n_tail:i,n], ydata[i-n_tail:i,n])
      else:
        self.lines_plt[n].set_data(xdata[0:i,n], ydata[0:i,n])
    
    # Pair data
    p_obs, p_rbt = [xdata[i,1], ydata[i,1]], [xdata[i,0], ydata[i,0]]
    p_rel = np.array(p_obs) - np.array(p_rbt)

    v_obs, v_rbt = self.v[1], self.v[0]
    phi_rbt = phidata[i,0]
    phi_obs = phidata[i,1]

    v_rel = (v_obs*np.array([np.cos(phi_obs), np.sin(phi_obs)]) - \
              v_rbt*np.array([np.cos(phi_rbt), np.sin(phi_rbt)]) )/4

    # GRAY CONES
    p1, p2 = cone_params(p_obs, rho_val[i,0,1], p_rbt)

    self.red_cone.remove()
    self.red_cone = patches.Polygon(np.array([p_rbt, p1, p2]),
                          alpha=0.05, color="gray", zorder=1, lw=0)
    self.anim_axis.add_patch(self.red_cone)
    

    # RED CONE
    p1, p2 = cone_params(p_obs - 2*p_rel, rho_val[i,0,1], p_rbt)
    self.gray_cone.remove()
    self.gray_cone = patches.Polygon(np.array([p_rbt, p1, p2]),
                          alpha=0.2, color="red", zorder=1, lw=0)
    self.anim_axis.add_patch(self.gray_cone)

    # Drawing vectors
    self.prel_quiv.set_offsets([p_rbt[0], p_rbt[1]])
    self.prel_quiv.set_UVC(p_rel[0], p_rel[1])
    self.vrel_quiv.set_offsets([p_rbt[0], p_rbt[1]])
    self.vrel_quiv.set_UVC(v_rel[0], v_rel[1])

    # Virtual collision zone
    self.icon_real.remove()
    self.icon_real = patches.Circle(p_obs, self.r, alpha=0.4, fill=True,
                          color=COLOR_PLT[1], ls="-", lw=0, zorder=0)
    self.anim_axis.add_patch(self.icon_real)

    self.icon_virt.remove()
    self.icon_virt = patches.Circle(p_obs, rho_val[i,0,1], alpha=0.2, fill=True,
                          color="gray", ls="-", lw=0, zorder=-1)
    self.anim_axis.add_patch(self.icon_virt)

    # Data vertical axis
    self.pline.set_xdata(i*self.dt/1000)
    self.lline.set_xdata(i*self.dt/1000)
    self.wline.set_xdata(i*self.dt/1000)

  
  """\
  Funtion to generate the full animation of the simulation
  """
  def generate_animation(self, output_folder, tf_anim=None, res=1920, n_tail=200):
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
    lgh      = self.data["lgh"]
    rho_val  = self.data["rho_val"]
    prelvi   = self.data["prelvi"]
    
    # -- Generating the animation --
    # Figure and grid init
    figsize = (16,9)

    fig = plt.figure(figsize=figsize, dpi=res/figsize[0])
    grid = plt.GridSpec(3, 5, hspace=0.1, wspace=0.4)

    self.anim_axis  = fig.add_subplot(grid[:, 0:3])
    prel_ax  = fig.add_subplot(grid[0, 3:5], xticklabels=[])
    lgh_ax = fig.add_subplot(grid[1, 3:5], xticklabels=[])
    prelvi_ax = fig.add_subplot(grid[2, 3:5])

    # Axis formatting
    self.anim_axis.set_xlim(PX_LIMS)
    self.anim_axis.set_ylim(PY_LIMS)
    self.anim_axis.set_ylabel(r"$p_y$ (L)")  
    self.anim_axis.set_xlabel(r"$p_x$ (L)")
    self.anim_axis.grid(True)

    fmt_data_axis(prel_ax, ylabel = r"$||p_{ij}||$ [L]")
    fmt_data_axis(lgh_ax, r"$L_gh^i$", ylim=LDATA_LIMS)
    fmt_data_axis(prelvi_ax, r"$\hat p_{ij}^\top E \hat v_i$", r"$t$ (T)", ylim=[-1.1,1.1])

    # -- Main axis
    self.gvf_traj.draw(fig, self.anim_axis, width=0.0024, alpha=0.1, lw=2)

    p_obs, p_rbt = [xdata[0,1], ydata[0,1]], [xdata[0,0], ydata[0,0]]
    p_rel = np.array(p_obs) - np.array(p_rbt)

    # Icon
    self.icons_plt = []

    for n in range(self.sim.N):
      icon = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], COLOR_PLT[n], size=1.2)
      self.anim_axis.add_patch(icon)
      self.icons_plt.append(icon)

      # GRAY CONES
      p1, p2 = cone_params(p_obs, rho_val[0,0,1], p_rbt)
      self.gray_cone = patches.Polygon(np.array([p_rbt, p1, p2]),
                            alpha=0.05, color="gray", zorder=1, lw=0)
      self.anim_axis.add_patch(self.gray_cone)

      # RED CONE
      p1, p2 = cone_params(p_obs - 2*p_rel, rho_val[0,0,1], p_rbt)
      self.red_cone = patches.Polygon(np.array([p_rbt, p1, p2]),
                            alpha=0.2, color="red", zorder=1, lw=0)
      self.anim_axis.add_patch(self.red_cone)

    # Drawing vectors
    v_obs, v_rbt = self.v[1], self.v[0]
    phi_rbt = phidata[0,0]
    phi_obs = phidata[0,1]
    v_rel = (v_obs*np.array([np.cos(phi_obs), np.sin(phi_obs)]) - \
            v_rbt*np.array([np.cos(phi_rbt), np.sin(phi_rbt)]) )/4

    self.vrel_quiv = self.anim_axis.quiver(p_rbt[0], p_rbt[1], v_rel[0], v_rel[1], **ARR_KW, color="red")
    self.prel_quiv = self.anim_axis.quiver(p_rbt[0], p_rbt[1], p_rel[0], p_rel[1], **ARR_KW, color="k")

    # Virtual collision zone
    self.icon_real = patches.Circle(p_obs, self.r, alpha=0.4, fill=True,
                              color=COLOR_PLT[1], ls="-", lw=0, zorder=0)
    self.anim_axis.add_patch(self.icon_real)

    self.icon_virt = patches.Circle(p_obs, rho_val[0,0,1], alpha=0.2, fill=True,
                              color="gray", ls="-", lw=0, zorder=-1)
    self.anim_axis.add_patch(self.icon_virt)

    # Trace
    self.lines_plt = []
    for n in range(self.sim.N):
      line, = self.anim_axis.plot(xdata[0:0,n], ydata[0:0,n], c=COLOR_LINE_PLT[n], ls="-", lw=1.2, zorder=0)
      self.lines_plt.append(line)

    # -- Data axis
    time_vec = np.linspace(0, self.tf/1000, int(self.tf/self.dt))
    l0 = 2

    # Zero lines
    prel_ax.axhline(self.r,  c="black", ls="--", lw=1.2, zorder=0, alpha=1)
    lgh_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=0.8)
    prelvi_ax.axhline(0, c="black", ls="--", lw=1.2, zorder=0, alpha=0.8)

    prel_ax.axvline(0,  c="black", ls="-", lw=1.2, zorder=0, alpha=1)
    lgh_ax.axvline(0, c="black", ls="-", lw=1.2, zorder=0, alpha=1)
    prelvi_ax.axvline(0, c="black", ls="-", lw=1.2, zorder=0, alpha=1)

    # Plotting data
    prelvi_ax.plot(time_vec[l0:], prelvi[l0:,1,0], c=COLOR_RBT, lw=1.2, alpha=1)

    for n in range(self.sim.N):
      for k in range(self.sim.N):
        if k > n:
          prel_ax.plot(time_vec[l0:], preldata[l0:,k,n], c=COLOR_RBT, lw=1.2, alpha=1, zorder=2)
          if self.v[n] > self.v[k]:
            lgh_ax.plot(time_vec[l0:], lgh[l0:,k,n], c=COLOR_RBT, lw=1.2, alpha=1, zorder=2)

    self.pline = prel_ax.axvline(0, c="black", ls="--", lw=1.2)
    self.lline = lgh_ax.axvline(0, c="black", ls="--", lw=1.2)
    self.wline = prelvi_ax.axvline(0, c="black", ls="--", lw=1.2)

    # Legend settings
    self.red_cone.set_label("Col. cone")
    self.icon_real.set_label("Real col. zone")
    self.icon_virt.set_label("Virtual col. zone")
    self.anim_axis.legend(loc="upper left", ncol=3, fancybox=True, framealpha=1, fontsize=13)

    # -- Animation --
    # Init of the animation class
    anim = FuncAnimation(fig, self.animate, fargs=(xdata, ydata, phidata, rho_val, n_tail),
                         frames=tqdm(range(frames), initial=1, position=0), interval=1/fps*1000)

    # Generate and save the animation
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Jesús Bautista Villar'), bitrate=10000)
    anim.save(os.path.join(output_folder, "anim__{0}_{1}_{2}__{3}_{4}_{5}.mp4".format(*time.localtime()[0:6])), 
              writer = writer)