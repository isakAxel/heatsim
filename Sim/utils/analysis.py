import os
import sys
import numpy as np

import Sim.heatsim2d as s
import pandas as pd


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

        


n = 11
T_end = 1000
c_analysis(n,T_end)