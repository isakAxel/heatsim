"""
    Module for 3D-Simulations (WIP) . The module contains 
"""
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import sparse, linalg, special
from scipy.sparse.linalg import spsolve, splu,cg, gmres

from Sim.config import build_sim,SimulationSetup
from Sim.operators import DiscLaplacian1D, F1, dF1,BoundaryConditionNeumann, F2_xyz, lambda_xyFFT,fft_apply
from Sim.utils.profiling import timer,cpuprf
from Sim.utils.utils import save,readh5
import Sim.visual as vis
from Sim.solvers import thomasMethodJAX,thomasMethodJ,thomasMethodJ2
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
    return np.asarray(u)

def IkFLz3D_JAX(v, k, Lzs, fq, cfg):

    
    Mz2,My, Mx = cfg.space.Mz + 2,cfg.space.My ,cfg.space.Mx
    idx = fq.nonzero()[0]
    w = np.sqrt(cfg.space.hx * cfg.space.hz * cfg.space.hy )
    # diags are dense
    lower = jnp.asarray(-k/2 * Lzs.diagonal(-1))
    upper = jnp.asarray(-k/2 * Lzs.diagonal(1) ) 
    dia  = 1.0 - k/2 * Lzs.diagonal(0)
    dia = jnp.tile(dia,Mx*My)

    u = v.copy()
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    for i in range(cfg.time.maxit):

        ### scipy sparse matrices, but really only scalar entries
        Fu  = F1(u, cfg, fq).data
        dFu = dF1(u, cfg, fq).data

        dia_i = dia.copy()
        dia_i = dia_i.at[idx].add(-k/2 * dFu)
        
        U = u.reshape(Mx*My, Mz2)
        Lu = (Lzs @ U.T).T.ravel()       # apply Lzs along z to each line

        r = u - k/2* Lu - v # apply the 1D operator along z
        # r[idx] -= k/2 * Fu
        r = r.at[idx].add( -k/2 * Fu)

        du = thomasMethodJ(lower,dia_i,upper,r,Mx,My,Mz2)
        u -= du

        if w * np.linalg.norm(du) < cfg.time.tol:
            break
        if i == (cfg.time.maxit - 1):
            print("Did not convergence iteration = ", i)
    return np.asarray(u)

@timer()
@cpuprf
def PRCoolingCycleDS_Naive(v0,Lx,Ly,Lzs, fq, cfg,history = 0, comp = "N"):
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

    # To compare methods compiling methods JAX AND NUMBA
    if comp == "J":
        NewtonIter = IkFLz3D_JAX
    else:
        NewtonIter = IkFLz3D_Numba 

    for n in range(N):
        # print("iteration", n)
        wnew = wn + k/2 * (Lzb @ wn)
        wnew[idx] += k/2 * F1(wn, cfg, fq).data
        # TODO: can this be precomputed since (I+ k/2LxLy)(I-k/2LxLy)^-1 is constant throughout iterations N
        # (I + k/2 L)(I - k/2 L)^-1
        v = solve_cg(A, wnew, x0 = None, rtol = cfg.time.tol * 1e-5)
        v = (I + k/2 * L) @ v 
    
        # (I - k/2 Lz - k/2 F)^-1
        wn = NewtonIter(v,k,Lzs,fq,cfg)

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

def PRCoolingCycleDS_FFT(v0,Lx,Ly,Lzs,fq, cfg, history = 0, comp = "N"):
    """
        Dimensional Split with FFT-in-plane:
            Experimental method that uses intial strang splitting for the exact operator,
            followed by a mixed FFT and trapezoidal step update. FFT in x-,y-directions, 
            and trapezoidal method for Z-line.

            u_n+1 = FFT(k/2 *v0,(x,y))(I - k/2*Fz)^-1(I + k/2*Fz) FFT(k/2 *v0,(x,y))
            u_n+1 = 
            Lx and Ly are "elevated" 3d Matrices, and Lzs is small
    """
    # Extracting parameters
    N = cfg.time.N
    k = cfg.time.k_pause
    Mx,My,Mz = cfg.space.Mx,cfg.space.My,cfg.space.Mz
    hx,hy = cfg.space.hx,cfg.space.hy
    D = cfg.target.D

    Lzb = sparse.kron(sparse.kron(sparse.eye_array(Mx, format="csr"), sparse.eye_array(My, format="csr"), format="csr"), Lzs, format="csr")

    print("------ Cooling Cycle Dimensional Split - Strang(FDM) with FFT  --------")
    if history:
        w = np.zeros((Mx*My*(Mz+2), N + 1))
        w[:, 0] = v0.copy()

    Vn = v0.reshape(Mx,My,Mz).copy()
    idx = fq.nonzero()[0]

    # Fourier modes for periodic laplacian
    Lx,Ly = np.zeros(Mx),np.zeros(My)
    Lx[0],Ly[1],Ly[-1] = -2,1,1
    Ly[0],Ly[1],Ly[-1] = -2,1,1
    # comput eigenvalues
    lam_xy_elev = lambda_xyFFT(Lx, hx, Ly, hy)[:, :, None] 
    for n in range(N):
        if n == 0:
            # Returns into spatial domain
            Vn = fft_apply(Vn,Mx,My,D,k/2)

        # Apply Exp step
        vn = Vn.reshape(-1)
        vn2 = vn + k/2*(Lzb@vn)
        vn2[idx] += k/2 * F1(vn,cfg,fq).data

        # Apply Implicit step
        vn2 = IkFLz3D_Numba(vn2,k,Lzs,fq,cfg)
        Vn = vn2.reshape(Mx,My,Mz)

        if n == (N - 1):
            V_final = fft_apply(Vn,Mx,My,D,k/2) 
        else:
            Vn = fft_apply(Vn,Mx,My,D,k)    
    return V_final.reshape(-1)

def _line_tridiag(L, stride, n):
    """
    Tridiagonal (lower, diag, upper) of the 1D operator inside an elevated Kronecker operator.
    stride : flat-index distance between neighbouring points along the line
             (x: My*(Mz+2), y: Mz+2)
    n      : number of points along the line
    """
    lo = L.diagonal(-stride)[::stride][:n-1]
    d  = L.diagonal(0)[::stride][:n]
    up = L.diagonal(stride)[::stride][:n-1]
    return lo, d, up


def _xy_line_solvers(Lx, Ly, c, Mx, My, Mz2):
    """
    Returns solve_x, solve_y with  solve_x(b) = (I - c*Lx)^-1 b  and  solve_y(b) = (I - c*Ly)^-1 b,
    for vectors in the ordering index = (ix*My + iy)*Mz2 + iz. Uses thomasMethodJAX.
    """
    lo, d, up = _line_tridiag(Lx, My*Mz2, Mx)
    lx, dx, ux = -c*lo, np.tile(1.0 - c*d, My*Mz2), -c*up      # lines along x: My*Mz2 of them
    lo, d, up = _line_tridiag(Ly, Mz2, My)
    ly, dy, uy = -c*lo, np.tile(1.0 - c*d, Mx*Mz2), -c*up      # lines along y: Mx*Mz2 of them

    def solve_x(b):
        B = np.ascontiguousarray(b.reshape(Mx, My, Mz2).transpose(1, 2, 0)).ravel()   # x fastest
        s = thomasMethodJAX(lx, dx, ux, B, My, Mz2, Mx)
        return s.reshape(My, Mz2, Mx).transpose(2, 0, 1).ravel()

    def solve_y(b):
        B = np.ascontiguousarray(b.reshape(Mx, My, Mz2).transpose(0, 2, 1)).ravel()   # y fastest
        s = thomasMethodJAX(ly, dy, uy, B, Mx, Mz2, My)
        return s.reshape(Mx, Mz2, My).transpose(0, 2, 1).ravel()

    return solve_x, solve_y


@timer()
@cpuprf
def PRCoolingCycleDS_3split_Thomas(v0, Lx, Ly, Lzs, fq, cfg, history=0):
    """
        Dimensional--Triple Split into Lx / Ly / (Lz + Fq):
            Experimental method that uses intial strang splitting for the exact operator,
            followed by a trapezoidal rule update, to hopefully retain order 2 convergance

            u_n+1 = (I - k/2 (Lz+F))^-1 (I + k/2 LxLy)(I - k/2 LxLy)^-1 (I + k/2 (Lz+F)) u_n
            Lx and Ly are "elevated" 3d Matrices, and Lzs is small
            The Lx / Ly systems are tridiagonal along their own axis and solved with the Thomas method.
    """
    N = cfg.time.N
    k = cfg.time.k_pause
    Mx, My, Mz = cfg.space.Mx, cfg.space.My, cfg.space.Mz
    Mz2 = Mz + 2

    Lzb = sparse.kron(sparse.kron(sparse.eye_array(Mx, format="csr"), sparse.eye_array(My, format="csr"), format="csr"), Lzs, format="csr")

    M3 = Lx.shape[0]
    wn = v0.copy()
    idx = fq.nonzero()[0]
    I = sparse.eye_array(M3, M3, format="csr")
    if history:
        w = np.zeros((M3, N + 1))
        w[:, 0] = v0.copy()

    print("------ Cooling Cycle Dimensional Split - Triple Split FDM --------")
    Y = (I + k/4*Ly)
    X = (I + k/4*Lx)
    solve_Yi, solve_Xi = _xy_line_solvers(Lx, Ly, k/4, Mx, My, Mz2)   # (I - k/4 Ly)^-1, (I - k/4 Lx)^-1

    for n in range(N):
        # Apply S
        wn = solve_Yi(Y @ wn)
        wn = solve_Xi(X @ wn)
        # Nonlinear part
        wnew = wn + k/2 * (Lzb @ wn)
        wnew[idx] += k/2 * F1(wn, cfg, fq).data
        # Nonlinear inverse
        wn = IkFLz3D_Numba(wnew, k, Lzs, fq, cfg)
        # Apply 2nd S
        wn = solve_Yi(Y @ wn)
        wn = solve_Xi(X @ wn)

        if history:
            w[:, n+1] = wn
    return w if history else wn


def HeatIter(cfg : SimulationSetup,Lx_s,Ly_s,Lz_s,fq_s,plot_option = False,save_path = None):
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
    #Show bragg peak
    if plot_option:
        dz = cfg.target.dEdz[0]
        dE = cfg.target.dEdz[1]
        label = [f"Ep= {cfg.beam.Ep}"]
        vis.plot_Ds(dz,[dE],label)
        vis.plot_3D(cfg,F2_xyz_MC(cfg) * cfg.beam.t_pulse,"Temperature after 1 heating cycle","MC")
    #unpacking stuff
    p = cfg.beam.p
    t_pause, t_pulse = cfg.beam.t_pause,cfg.beam.t_pulse
    Mx,My,Mz = cfg.space.Mx,cfg.space.My,cfg.space.Mz
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
    # Heatin cycle contribution constant over pulses -> precompute
    HC = F2_xyz_MC(cfg) * cfg.beam.t_pulse
    # Iterate pulses
    for pk in range(p):
        # Heating Cycle
        v += HC
        maxT.append(np.max(v))
        ts.append(pk*(t_pulse+t_pause) + cfg.beam.t_pulse)
        # Cooling Cycle
        v = PRCoolingCycleDS_3split_Thomas(v,Lx,Ly,Lz_s, Fq, cfg,history = 0)
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

    if save_path:
        save(cfg,v,maxT,ts,save_path,name = f"Sim_{cfg.target.material}")
        
    return v,maxT,ts

if __name__ == "__main__":
        """
        Scripting interface as of now, write things in main to run the simulation
        """
        
        plot_option = 1
        # Resolution parameters
        M = 50 
        N = 50
        Mz = My = Mx = M
        ##  Building the parameter, default parameters are listed in config.py
        cfg = build_sim(N=N, Mx=M, Mz=M, My = M, t_pulse=1.5e-3,sigmax=2,sigmay=2,p = 5, Ep = 800)
    
        print("---------- Parameters for Simulation -------------\n",
              f"Simulation Material : {cfg.target.material} \n",
              f"Pulse period : {1/cfg.beam.p_freq} \n",
              f"Pulse frequency : {cfg.beam.p_freq:.2f} \n",
              f"Pulse length : {cfg.beam.t_pulse} \n",
              f"Energy : {cfg.beam.Ep} \n",
              f"Time without pulse (Pause length): {cfg.time.t_pause}\n",
              f"Density rho : {cfg.target.rho}\n",
              f"Specific Heat Cp: {cfg.target.C_p}\n",
              f"Stopping power dEdz: {cfg.target.dEdz[1][0]:.2E}\n",
              f"Peak current Ib : {cfg.beam.Ib}\n",
              f"Sigmax : {cfg.beam.sigmax} \n",
              f"Sigmay : {cfg.beam.sigmay} \n",
              f"T0 : {cfg.T0}\n")

        # --- Operators ---------------------------------------------------
        # Laplacians
        Lx_s = DiscLaplacian1D(cfg, axis="x")
        Ly_s = DiscLaplacian1D(cfg, axis="y")
        Lz_s = DiscLaplacian1D(cfg, axis="z")
        # Adding neumann condition
        Lz_s, fq1d = BoundaryConditionNeumann(Lz_s, q0=1, qend=2)
        
        # vis.plot_3D(cfg,v0, f"Initial temperature after heating cycle", "-")
        u = PRCoolingCycleDS_Naive(v0,Lx,Ly,Lz_s, Fq, cfg,history = 0, comp= "N", cpuprf = True)
        u2 = PRCoolingCycleDS_FFT(v0,Lx,Ly,Lz_s, Fq, cfg,history = 0, comp= "N",cpuprf = True)
        
        # u2 = PRCoolingCycleDS_3split(v0,Lx,Ly,Lz_s, Fq, cfg,history = 0, timeit = False)
        # u = PRCoolingCycleDS_Naive(v0,Lx,Ly,Lz_s, Fq, cfg,history = 0,timeit = False)

        u_diff = u - u2

        # Plotting max slice       
        if plot_option:
            u = v.reshape(Mx * My, Mz + 2)        # index = (ix*My + iy)*(Mz+2) + iz
            # Maximum over the xy-plane for each z-slice, then the z-slice with the largest one
            slice_max = u.max(axis=0)
            s1 = int(np.argmax(slice_max))
            u1 = u[:, s1].reshape(Mx, My)

            print(f"max slice index s1 = {s1}, depth z = {np.linspace(0, cfg.target.Lz, Mz + 2)[s1]:.4f} cm")

            vis.plot_2D_field(cfg, u1, f"Heat after 1st cooling cycle for slice (i_z={s1}),\n M = {Mx}, N = {N}. ", "PR-DS3D",
                                mark_max=True)
            
