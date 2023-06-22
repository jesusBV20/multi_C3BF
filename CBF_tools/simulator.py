import numpy as np

E = np.array([[0, 1],[-1, 0]])

# ----------------------------------------------------------------------
# Dinámica de un rover con volante (velocidad constante)
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

# -> x0:    [t0, p0, v0, phi0]

# - P es una matriz  N x 2
# - v es un vector   N x 1 (norma de la velocidad)
# - phi es un vector N x 1
# - w es un vector   N x 1

class simulator:
    def __init__(self, gvf_traj, n_agents, x0, dt = 0.01, cbf_sw = False, kinematics=rover_kinematics):
      self.traj = gvf_traj         # Trayectoria a seguir por los agentes
      self.kinematics = kinematics # Dinámica de los robots
      self.N = n_agents            # Número de agentes simulados
      self.t0 = x0[0]              # Tiempo inicial de la simulación (s)
      self.t  = x0[0]              # Tiempo actual de la simulación (s)
      self.dt = dt                 # Pasos temporales en la simulación (s)

      # Vectores de estados (fila -> agente)
      self.p0 = x0[1]    # Matriz N x 2 de posiciones iniciales
      self.pf = x0[1]
      self.v0 = x0[2]
      self.vf = x0[2]
      self.phi0 = x0[3]
      self.phif = x0[3]
      self.w = np.zeros([n_agents])
      self.e = np.zeros([n_agents])

      # Parámetros del robot
      self.L = 0.8                      # m
      self.delta_max = 15 * np.pi / 180 # rad

      # Parámetros de los controladores
      self.s  = 1
      self.ke = 1
      self.kn = 1

      self.r = 0.6
      self.gamma = 1

      # Switch para seleccionar el tipo de algoritmo para evitar colisiones
      self.cbf_sw = cbf_sw

      # Declaramos variables a monitorizar ---------------------
      self.p_rel = np.zeros([n_agents, n_agents, 2])
      self.v_rel = np.zeros([n_agents, n_agents, 2])
      self.h = np.zeros([n_agents, n_agents])
      self.h_dot = np.zeros([n_agents, n_agents])
      self.kappa = np.zeros([n_agents, n_agents])
      self.psi = np.zeros([n_agents, n_agents])

      self.omega_safe = np.zeros([n_agents])
      # --------------------------------------------------------

    def set_params(self, s = None, ke = None, kn = None, r = None, gamma = None):
      params = [s, ke, kn, r, gamma]
      params_def = self.s, self.ke, self.kn, self.r, self.gamma

      for i in range(len(params)):
        if params[i] == None:
          params[i] == params_def[i]

      self.s, self.ke, self.kn, self.r, self.gamma = params

    def omega_clip(self, omega):
      max_w = self.vf/self.L * np.tan(self.delta_max) # N x 1
      w_clipped = omega
      for i in range(self.N):
        w_clipped[i] = np.clip(omega[i], -max_w[i], max_w[i])
      return w_clipped

    def gvf_control(self):
      # Phi (error)  N x 1
      self.e = self.traj.phi(self.pf)
      # Phi gradient N x 2
      n = self.traj.grad_phi(self.pf)
      # Phi tangent  N x 2
      t = self.s*n @ E.T
      # Phi hessian  2 x 2
      H = self.traj.hess

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
      omega = self.omega_clip(omega)

      return -omega

    def cbf_colAvoid(self):
      omega = np.copy(self.w)

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
        omega_ref = self.w[i]
        P = self.pf
        V = self.vf * np.array([np.cos(self.phif), np.sin(self.phif)]).T

        psi_lgh_k = []
        for k in [k for k in range(self.N) if k!=i]:
          v = self.vf[i]
          phi = self.phif[i]

          prel = P[i,:] - P[k,:]
          prel_sqr = np.dot(prel, prel)
          prel_norm = np.sqrt(prel_sqr)

          if prel_sqr > self.r**2: # Si no ha colisionado ...
            cos_alfa = np.sqrt(prel_sqr - self.r**2)/prel_norm

            vrel = V[i,:] - V[k,:]
            vrel_norm = np.sqrt(np.dot(vrel, vrel))
            vrel_dot_g = v * np.array([-np.sin(phi), np.cos(phi)])

            # h(x,t)
            dot_rel = np.dot(prel, vrel) # TODO: plot this
            h = dot_rel + prel_norm * vrel_norm * cos_alfa

            # h_dot_r(x,t) = h_dot(x, u_ref(x,t))
            vrel_dot_r = vrel_dot_g * omega_ref
            h_dot_r = vrel_norm**2 + np.dot(prel, vrel_dot_r) + \
                            np.dot(vrel, vrel_dot_r)*(cos_alfa*prel_norm)/vrel_norm + \
                            dot_rel*vrel_norm/(cos_alfa*prel_norm)

            # psi(x,t)
            psi = h_dot_r + self.gamma * h

            # Lgh = grad(h(x,t)) * g(x) = dh/dvrel_dot * vrel_dot
            Lgh = np.dot(prel + vrel * (cos_alfa*prel_norm)/vrel_norm, vrel_dot_g)

            # Explicit solution of the QP problem
            if psi < 0:
              psi_lgh_k.append(- psi / Lgh)

            # Volcamos los resultados sobre las variables de telemetría --
            self.p_rel[k,i] = prel
            self.v_rel[k,i] = vrel
            self.h[k,i] = h
            self.h_dot[k,i] = h_dot_r
            self.kappa[k,i] = self.gamma * h
            self.psi[k,i] = psi

            # ------------------------------------------------------------

        if len(psi_lgh_k) != 0:
          if self.s == 1:
            self.omega_safe[i] = np.max([np.max(psi_lgh_k), 0])
          else:
            self.omega_safe[i] = np.min([np.min(psi_lgh_k), 0])
          omega[i] += self.omega_safe[i]

      # Clip the response of the controller by the mechanical saturation value
      omega = self.omega_clip(omega)
      return omega

    def int_euler(self):
      self.w = self.gvf_control()
      if self.cbf_sw:
        self.w = self.cbf_colAvoid()
      [p_dot, v_dot, phi_dot] = self.kinematics(self.pf, self.vf, self.phif, self.w)
      self.t = self.t + self.dt
      self.pf = self.pf + p_dot * self.dt
      self.vf = self.vf + v_dot * self.dt
      self.phif = (self.phif + phi_dot * self.dt) % (2*np.pi)