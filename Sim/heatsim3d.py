"""
    Module for 3D-Simulations (WIP) . The module contains 
"""
import numpy as np
from scipy import sparse, linalg, special
from scipy.sparse.linalg import spsolve, splu,cg, gmres

from Sim.config import build_sim,SimulationSetup
from Sim.operators import DiscLaplacian1D, F1, dF1,BoundaryConditionNeumann, F2_xyz
from Sim.utils.profiling import timer,cpuprf
import Sim.visual as vis
from Sim.solvers import thomasMethodJAX
# Numerically stable sigmoid
expit = special.expit

from pypardiso import PyPardisoSolver


def IkFLz3D_Numba(v, k, Lzs, fq, cfg):

    
    Mz2,My, Mx = cfg.space.Mz + 2,cfg.space.My ,cfg.space.Mx
    idx = fq.nonzero()[0]
    w = np.sqrt(cfg.space.hx * cfg.space.hz * cfg.space.hy )
    # diags are dense
    lower = -k/2 * Lzs.diagonal(-1)
    upper = -k/2 * Lzs.diagonal(1)
    dia  = 1.0 - k/2 * Lzs.diagonal(0)
    dia = np.tile(dia,Mx*My)

    u = v.copy()
    for i in range(cfg.time.maxit):

        ### scipy sparse matrices, but really only scalar entries
        Fu  = F1(u, cfg, fq).data
        dFu = dF1(u, cfg, fq).data

        dia_i = dia.copy()
        dia_i[idx] -= k/2 * dFu

        U = u.reshape(Mx*My, Mz2)
        Lu = (Lzs @ U.T).T.ravel()       # apply Lzs along z to each line

        r = u - k/2* Lu - v # apply the 1D operator along z
        r[idx] -= k/2 * Fu
       
        # No difference between SymmSolver and ThomasMethodJax
        du = thomasMethodJAX(lower,dia_i,upper,r,Mx,My,Mz2)
        u -= du

        if w * np.linalg.norm(du) < cfg.time.tol:
            break
        if i > 99:
            print("Iterations:", i)
    return u

@timer()
@cpuprf
def PRCoolingCycleDS_Naive(v0,Lx,Ly,Lzs, fq, cfg,history = 0):
    """
    Dimensional splitting  Lx + Ly  /  (Lz + F1_q):

        u_n+1 = (I - k/2 (Lz+F))^-1 (I + k/2 LxLy)(I - k/2 LxLy)^-1 (I + k/2 (Lz+F)) u_n
        Lx and Ly are "elevated" 3d Matrices, and Lzs is small
    """
    N = cfg.time.N
    k = cfg.time.k_pause
    Mx,My,Mz = cfg.space.Mx,cfg.space.My, cfg.space.Mz
        
    
    Lzb = sparse.kron(sparse.kron(sparse.eye_array(Mx, format="csr"), sparse.eye_array(My, format="csr"), format="csr"), Lzs, format="csr")
    
    M3 = Lx.shape[0]
    wn = v0.copy()    
    idx = fq.nonzero()[0]
    I = sparse.eye_array(M3,M3, format = "csr")
    if history:
        w = np.zeros((M3, N + 1))
        w[:, 0] = v0.copy()

    print("------ Cooling Cycle Dimensional Split --------")
    L = Lx + Ly
    A = (I - k/2 * L)

    for n in range(N):

        wnew = wn + k/2 * (Lzb @ wn)
        wnew[idx] += k/2 * F1(wn, cfg, fq).data
        # TODO: can this be precomputed since (I+ k/2LxLy)(I-k/2LxLy)^-1 is constant throughout iterations N
        # (I + k/2 L)(I - k/2 L)^-1
        v = solve_cg(A, wnew, x0 = None, rtol = 1e-8)
        v = (I + k/2 * L) @ v 
    
        # (I - k/2 Lz - k/2 F)^-1
        wn = IkFLz3D_Numba(v,k,Lzs,fq,cfg)

        if history:
            w[:, n + 1] = wn
    return w if history else wn

def solve_cg(A,b,rtol=1e-5,**kwargs):
    """
        Conjugate gradient solver that checks whether or not the cg iteration converged.
        rtol is specified for relative tolerance of convergence
    """

    X,info = cg(A,b,rtol = rtol, **kwargs)
    if info > 0:
        print(f"CG did not converge (info = {info})")
    return X

@timer()
@cpuprf 
def PRCoolingCycleDS_3split(v0,Lx,Ly,Lzs,fq, cfg, history = 0):
    """
        Dimensional--Triple Split into Lx / Ly / (Lz + Fq):
            Experimental method that uses intial strang splitting for the exact operator,
            followed by a trapezoidal rule update, to hopefully retain order 2 convergance
    
            u_n+1 = (I - k/2 (Lz+F))^-1 (I + k/2 LxLy)(I - k/2 LxLy)^-1 (I + k/2 (Lz+F)) u_n
            u_n+1 = 
            Lx and Ly are "elevated" 3d Matrices, and Lzs is small
    """
    N = cfg.time.N
    k = cfg.time.k_pause
    Mx,My,Mz = cfg.space.Mx,cfg.space.My, cfg.space.Mz
    
    Lzb = sparse.kron(sparse.kron(sparse.eye_array(Mx, format="csr"), sparse.eye_array(My, format="csr"), format="csr"), Lzs, format="csr")
    
    M3 = Lx.shape[0]
    wn = v0.copy()    
    idx = fq.nonzero()[0]
    I = sparse.eye_array(M3,M3, format = "csr")
    if history:
        w = np.zeros((M3, N + 1))
        w[:, 0] = v0.copy()

    print("------ Cooling Cycle Dimensional Split - Triple Split FDM --------")
    Yi = (I- k/4*Ly)
    Y = (I+ k/4*Ly)
    Xi = (I- k/4*Lx)
    X = (I+ k/4*Lx)

    for n in range(N):
        # Apply S
        wn = solve_cg(Yi,Y @ wn,rtol = 1e-8)
        wn = solve_cg(Xi,X @ wn,rtol = 1e-8)
        # Nonlinear part
        wnew = wn + k/2 * (Lzb @ wn)
        wnew[idx] += k/2 * F1(wn,cfg,fq).data
        # Nonlinear inverse
        wn = IkFLz3D_Numba(wnew,k,Lzs,fq,cfg)
        # Apply 2nd S
        wn = solve_cg(Yi,Y @ wn,rtol = 1e-8)
        wn = solve_cg(Xi,X @ wn,rtol = 1e-8)

        if history:
            w[:,n+1] = wn
    return w if history else wn

def HeatIter(cfg : SimulationSetup,Lx_s,Ly_s,Lz_s,fq_s,plot_option = False):
    """
        Function that runs the full simulation time T = p*(t_pause + t_pulse) i.e. p pulses with heating and cooling phase.

        Params:
            cfg (SimulationSetup) : Steup containing the simulation parameters
            Lx_s(sparray) : Spatial discretization (X-dir) 
            Ly_s(sparray) : Spatial discretization (Y-dir)
            Lz_s(sparray) : Spatial discretization (Z-dir)
            fq_s(sparray) : Boundary condition vector (Z-dir)
            plot_option(boolean) : Option to plot
    """
    
    #unpacking stuff
    p = cfg.beam.p
    t_pause, t_pulse = cfg.beam.t_pause,cfg.beam.t_pulse
    # Identities 
    Ix = sparse.eye_array(Mx,Mx,format = "csr")
    Iy = sparse.eye_array(My,My,format = "csr")
    Iz = sparse.eye_array(Mz+2,Mz+2,format = "csr")
    _onesx = sparse.csr_array(np.ones((Mx, 1)))
    # Elevate operators
    Fq  = sparse.kron(_onesx, sparse.kron(_onesx,fq_s), format="csr") 
    Ly = sparse.kron(Ix, sparse.kron(Ly_s, Iz),format = "csr")
    Lx = sparse.kron(Lx_s,sparse.kron(Iy,Iz), format = "csr")
    
    # Initial data
    v = np.zeros(Mx*My*(Mz+2))
    maxT = []
    ts = []
    # stopping tolerance for pulsing
    tol = 1e-1
    # Iterate pulses
    for pk in range(p):
        
        # Heating Cycle
        v += F2_xyz(cfg) * cfg.beam.t_pulse
        maxT.append(np.max(v))
        ts.append(pk*(t_pulse+t_pause) + cfg.beam.t_pulse)
        # Cooling Cycle
        v = PRCoolingCycleDS_Naive(v,Lx,Ly,Lz_s, Fq, cfg,history = 0)
        maxT.append(np.max(v))
        ts.append((pk+ 1)*(t_pulse+t_pause))

        if pk > 1:      # We have 2 values to compare
            # cmp values at ended cooling cycle
            t_diff = np.abs(maxT[2*pk +1] - maxT[2*pk-1])

            if (t_diff < tol): 
                # Exit pulsing mode
                p = pk+1
                break


    
    if plot_option:
        
        vis.plot_T_evol(cfg,maxT,ts,title = f"Maximum Temperature evolution over {p} pulses", label = "Max Temperature", color = "red")
        vis.plot_3D(cfg,v,"Temperature after full Pulse cycle", "Using PR-DS w CG")

    return v,maxT,ts

if __name__ == "__main__":
        
        plot_option = 1
        M = 100 
        N = 100
        Mz = My = Mx = M
        cfg = build_sim(N=N, Mx=M, Mz=M, My = M, t_pulse=1.5e-3,sigmax=2,sigmay=2,p = 1)
        Mx, Mz, N = cfg.space.Mx, cfg.space.Mz, cfg.time.N
        h = cfg.space.hx
    
       
    
        print("---------- Parameters for Simulation -------------\n",
              f"Simulation Material : {cfg.target.material} \n",
              f"Pulse period : {1/cfg.beam.p_freq} \n",
              f"Pulse frequency : {cfg.beam.p_freq:.2f} \n",
              f"Pulse length : {cfg.beam.t_pulse} \n",
              f"Energy : {cfg.beam.Ep} \n",
              f"Time without pulse (Pause length): {cfg.time.t_pause}\n",
              f"Density rho : {cfg.target.rho}\n",
              f"Specific Heat Cp: {cfg.target.C_p}\n",
              f"Stopping power dEdz: {cfg.target.dEdz:.2E}\n",
              f"Peak current Ib : {cfg.beam.Ib}\n",
              f"Sigmax : {cfg.beam.sigmax} \n",
              f"Sigmay : {cfg.beam.sigmay} \n",
              f"T0 : {cfg.T0}\n")

        # --- Operators ---------------------------------------------------
        Lx_s = DiscLaplacian1D(cfg, axis="x")
        Ly_s = DiscLaplacian1D(cfg, axis="y")
        Lz_s = DiscLaplacian1D(cfg, axis="z")
        Lz_s, fq1d = BoundaryConditionNeumann(Lz_s, q0=1, qend=2)
        

        v,_,_ = HeatIter(cfg,Lx_s,Ly_s,Lz_s,fq1d,plot_option = True)

        # if False:
        #     vis.plot_3D(cfg,u_diff,f"3D distribution of heat after 1st cooling cycle M = {Mx}, N = {N}", "PR-DS_naive")
            
        if True:
            u = v.reshape(Mx*My,Mz+2)
            s1 = 34
            
            u1 = u[:,s1].reshape(Mx,My)

           
            vis.plot_2D_field(cfg, u1,
                f"Heat after 1st cooling cycle for slice (i_z={s1}),\n M = {Mx}, N = {N}. ",
                "PR-DS3D",
                mark_max=True)
            # vis.plot_2D_field(cfg, u2,
            #     f"Heat after 1st cooling cycle for slice (i_z={s2}),\n M = {Mx}, N = {N}. ",
            #     "PR-DS3d",
            #     mark_max=True)
