#!/usr/bin/env python #
"""\
# Copyright (C) 2023 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np

# Rotation matrix R(-pi/2)
E = np.array([[0, 1],[-1, 0]])

# ----------------------------------------------------------------------
# Steering rover dynamics (constant speed)
# ----------------------------------------------------------------------
def rover_kinematics(P, v, phi, w, m=1, f=[0,1,0]):
  Fr = lambda v : f[0] + f[1]*v + f[2]*v**2
  p_dot = v * np.array([np.cos(phi), np.sin(phi)]).T # Nx1 * Nx2 (dot product)
  v_dot = np.zeros(v.shape) # Constant speed
  phi_dot = w
  return p_dot, v_dot, phi_dot

# ----------------------------------------------------------------------
# GVF simulation class (gvf_control + cbf colision avoid controller)
# ----------------------------------------------------------------------

# -> x0:    [p0, v0, phi0]

# - P es una matriz  N x 2
# - v es un vector   N x 1 (norma de la velocidad)
# - phi es un vector N x 1
# - w es un vector   N x 1

class simulator:
  def __init__(self, gvf_traj, n_agents, x0, dt = 10, t_cbf = None, rho = None, rho_dot = None, obs = []):
    self.traj = gvf_traj               # Trayectory to be followed by the S-W-A-R-M
    self.kinematics = rover_kinematics # Robot dynamics integration function (Euler)
    self.N = n_agents                  # Total simulation agents (robots + obstacles)

    self.t, self.dt = 0, dt/1000 # Initial simulation time and time integration step (input in miliseconds -> to seconds)

    if t_cbf is None:
      self.t_cbf = 0
    else:
      self.t_cbf = t_cbf

    # State vector (row -> agente) --------------------
    self.pf = x0[0]    # N x 2 matrix  
    self.vf = x0[1]    # N vector
    self.phif = x0[2]  # N vector
    self.w = np.zeros([n_agents]) # N vector
    self.e = np.zeros([n_agents]) # N vector

    self.w_ref = np.zeros([n_agents])

    # Robot physical parameters to compute omega_max --
    self.L = 0.8                      # in meters
    self.delta_max = 15 * np.pi / 180 # in radians

    # Initial parameters of the GVF controller --------
    self.s  = 1
    self.ke = 1
    self.kn = 1

    # Initial parameters of the CBF controller --------
    self.r, self.gamma = 0.6, 1

    if rho is not None and rho_dot is not None: 
      self.rho = rho
      self.rho_dot = rho_dot
    else:
      self.rho = lambda prel,k: self.r
      self.rho_dot = lambda prel, vrel,k: 0

    self.obs = obs
    
    # Telemetry variables ------------------------
    self.p_rel = np.zeros([n_agents, n_agents, 2])
    self.v_rel = np.zeros([n_agents, n_agents, 2])
    self.h = np.zeros([n_agents, n_agents])
    self.h_dot = np.zeros([n_agents, n_agents])
    self.kappa = np.zeros([n_agents, n_agents])
    self.psi = np.zeros([n_agents, n_agents])

    self.omega_safe = np.zeros([n_agents])
    # ---------------------------------------------

  # -- Function to set the parameters of the simulations after the class construction
  def set_params(self, s = None, ke = None, kn = None, r = None, gamma = None):
    params = [s, ke, kn, r, gamma]
    params_def = self.s, self.ke, self.kn, self.r, self.gamma

    for i in range(len(params)):
      if params[i] == None:
        params[i] == params_def[i]

    self.s, self.ke, self.kn, self.r, self.gamma = params

  # -- Function to change the robot dynamics after the class construction
  def set_kinematics(self, kinematics_func):
    self.kinematics = kinematics_func

  # -- Function to bound the value of omega
  def omega_clip(self, omega):
    max_w = self.vf/self.L * np.tan(self.delta_max)
    w_clipped = omega
    for i in range(self.N):
      w_clipped[i] = np.clip(omega[i], -max_w[i], max_w[i])
    return w_clipped

  # -- Funtion to compute omega_gvf
  def gvf_control(self):
    # GVF trajectory data
    self.e = self.traj.phi(self.pf) # Phi (error)  N x 1
    n = self.traj.grad_phi(self.pf) # Phi gradient N x 2
    t = self.s*n @ E.T # Phi tangent  N x 2
    H = self.traj.hess # Phi hessian  2 x 2

    # Compute the desired angular velocity to aling with the vector field
    omega = np.zeros(self.N)
    for i in range(self.N):
      ei = self.e[i]
      ni = n[i,:][:,None]
      ti = t[i,:][:,None]

      pd_dot = ti - self.ke*ei*ni

      norm_pd_dot = np.sqrt(pd_dot[0]**2 + pd_dot[1]**2)
      md = pd_dot / norm_pd_dot

      Apd_dot_dot = - self.ke * ni.T @ pd_dot * ni
      Bpd_dot_dot = (E.T - self.ke * ei) @ H @ pd_dot
      pd_dot_dot = Bpd_dot_dot + Apd_dot_dot

      md_dot_const = md.T @ E.T @ pd_dot_dot / norm_pd_dot
      md_dot = E.T @ md * md_dot_const

      omega_d = md_dot.T @ E.T @ md
      mr = np.array([np.cos(self.phif[i]), np.sin(self.phif[i])])

      omega[i] = omega_d + self.kn * mr @ E.T @ md

    # Clip the response of the controller by the mechanical saturation value 
    # (YES, we can do this here)
    omega = self.omega_clip(omega)

    return -omega

  def cbf_colAvoid(self):
    omega = np.copy(self.w_ref)

    # Inicializamos las variables de telemetría
    self.p_rel = np.zeros(self.p_rel.shape)
    self.v_rel = np.zeros(self.v_rel.shape)
    self.h = np.zeros(self.h.shape)
    self.h_dot = np.zeros(self.h_dot.shape)
    self.kappa = np.zeros(self.kappa.shape)
    self.psi = np.zeros(self.psi.shape)
    self.omega_safe = np.zeros(self.omega_safe.shape)

    # Mientras se satisfacen las siguientes condiciones
    for i in range(self.N):
      P = self.pf
      V = self.vf * np.array([np.cos(self.phif), np.sin(self.phif)]).T

      v = self.vf[i]
      phi = self.phif[i]
      
      psi_lgh_k = []
      for k in [k for k in range(self.N) if k!=i]:
        # p_rel
        prel = P[k,:] - P[i,:]
        prel_sqr = np.dot(prel, prel)
        prel_norm = np.sqrt(prel_sqr)

        # v_rel
        vrel = V[k,:] - V[i,:]
        vrel_norm = np.sqrt(np.dot(vrel, vrel))

        if self.vf[k] < v:      
          phik = self.phif[k]
          if prel_norm > self.rho(prel,k): # Si no han colisionado ...
            cos_alfa = np.sqrt(prel_norm**2 - self.rho(prel,k)**2)/prel_norm

            # \dot v_rel (The -w is SO IMPORTANT)
            vrel_dot_1 = v * np.array([np.sin(phi), -np.cos(phi)])
            vrel_dot_2 = self.vf[k] * np.array([-np.sin(phik), np.cos(phik)])
            vrel_dot_ref = vrel_dot_2 * (self.w[k]) + vrel_dot_1 * (self.w_ref[i])

            # Derivative of terms involving A
            rho_rho_dot = self.rho(prel,k) * self.rho_dot(prel, vrel,k)

            # h(x,t)
            dot_rel = np.dot(prel, vrel)
            h = dot_rel + prel_norm * vrel_norm * cos_alfa

            # h_dot_ref(x,t) = h_dot(x, u_ref(x,t))
            h_dot_ref = vrel_norm**2 + np.dot(prel, vrel_dot_ref) + \
                            np.dot(vrel, vrel_dot_ref)*(cos_alfa*prel_norm)/vrel_norm + \
                            vrel_norm* (dot_rel - rho_rho_dot)/(cos_alfa*prel_norm)

            # psi(x,t)
            psi = h_dot_ref + self.gamma * h ** 3

            # Lgh = grad(h(x,t)) * g(x) = dh/dvrel_dot * vrel_dot
            Lgh = np.dot(prel + vrel * (cos_alfa*prel_norm)/vrel_norm, vrel_dot_1) \
                    
                  

            # Explicit solution of the QP problem
            delta = 0.1
            if abs(Lgh) > delta:
              if psi < 0:
                psi_lgh_k.append(- psi / Lgh) # This *(-) is SO IMPORTANT

            # CBF computation variables telemetry
            self.h[k,i] = h
            self.h_dot[k,i] = h_dot_ref
            self.kappa[k,i] = self.gamma * h
            self.psi[k,i] = psi
        
        # prel and vrel telemetry
        self.p_rel[k,i] = prel
        self.v_rel[k,i] = vrel

          # ------------------------------------------------------------

      if len(psi_lgh_k) != 0:
        if self.s == 1:
          self.omega_safe[i] = np.max([np.max(psi_lgh_k), 0])
        else:
          self.omega_safe[i] = np.min([np.min(psi_lgh_k), 0])
        omega[i] += self.omega_safe[i]

    # Clip the response of the controller by the mechanical saturation value (NOOO)
    #omega = self.omega_clip(omega)
    return omega

  def int_euler(self):
    self.w_ref = self.gvf_control()
    if self.t <= self.t_cbf:
      self.w = self.w_ref
      for i in self.obs:
        self.w[i] = 0
    else:
      self.w = self.cbf_colAvoid()
      for i in self.obs:
        self.w[i] = 0

    [p_dot, v_dot, phi_dot] = self.kinematics(self.pf, self.vf, self.phif, self.w)
    self.t = self.t + self.dt
    self.pf = self.pf + p_dot * self.dt
    self.vf = self.vf + v_dot * self.dt
    self.phif = (self.phif + phi_dot * self.dt) % (2*np.pi)