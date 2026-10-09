import os
import sys
import numpy as np

import Sim.heatsim2d as s
import pandas as pd
import Sim.operators as op
from Sim.heatsim2dMU import BoundaryConditionNeumann,F2_xz,F1
from scipy import sparse,linalg

def compute_coeffs(n_dU,T_end,cfg):

  # Extracting parameters
    t_pulse = cfg.time.t_pulse

    epsilon = cfg.target.epsilon
    dEdx = cfg.target.dEdz
    L = cfg.target.Lx
    D = cfg.target.D
    C_p = cfg.target.C_p
    rho = cfg.target.rho
    gamma = cfg.target.gamma
    A = cfg.target.A
    V = cfg.target.V
    sigma = cfg.sigma
    T0 = cfg.T0
    t = A/V
    Ib = cfg.beam.Ib
    # Testing for various \Delta U Values
    dU = np.linspace(0.0,float(T_end),n_dU)
    dU[0] += 1
    ### Could write these c_bolt and c_newton depending on \epsilon and \gamma ? 
    ### 
    print("tpulse = ", t_pulse)
    coeffs = {"dU" : dU.tolist(),
              "c_pulse" : (dEdx * Ib * L**2 / ( D**2*dU*C_p)).tolist(),
              "c_Bolt" : (epsilon * sigma * dU**3 * L ** 2 / (rho * C_p * t / D)).tolist(),
              "c_newton" :[ gamma*L**2 / (rho*C_p*t*D)] * len(dU),             # Does not depened on dU explicitly
              "Theta0" : (T0 / dU).tolist()
              }
    coeffs["c_laplace"] = [1]*len(dU) 
    return coeffs


def c_analysis(n_dU,T_end):

    coeffs = s.compute_coeffs(n_dU,T_end,s.par)
    df = pd.DataFrame(data=coeffs)

    
    print("------ Relative Coefficient values --------")
    print(df.to_string())


def check_comm(A,B,v0):
  """
  calculates the commutator between linear (matrix) A and nonlinear reaction term B

  Params:

  """


  comm = B(A@v0) - A @ B(v0)


  return comm

def TestComm():
  cfg = s.build_sim(t_pulse = 1.5e-3)
  Mx = Mz = cfg.space.Mx

  Ix = sparse.eye_array(Mx, Mx)
  Iz = sparse.eye_array(Mz + 2, Mz + 2)
  
  Lx = op.DiscLaplacian1D(cfg,"x")
  Lz = op.DiscLaplacian1D(cfg,"z")
  Lz, fq = BoundaryConditionNeumann(Lz,1,0)

  Lx = sparse.kron(Iz,Lx,format = "csr")
  Lz = sparse.kron(Lz,Ix,format = "csr") 

  fq = sparse.kron(fq,np.ones((Mx,1)) ,format = "csr")
  v0 = F2_xz(cfg) * cfg.beam.t_pulse

  def FqLz(v):
    
    idx = fq.nonzero()[0]
    k = cfg.time.k_pause

    vnew = v + k/2*(Lz@v)
    vnew[idx] += k/2*F1(v,cfg,fq).data

    return vnew

  comm = check_comm(Lx,FqLz,v0)
  print(f"Value of commutator : {comm}")
  normcomm = linalg.norm(comm,ord = 2)
  print(f"norm of commutator : {normcomm}")

if __name__ == '__main__':
  # n = 11
  # T_end = 1000
  # c_analysis(n,T_end)

  TestComm()