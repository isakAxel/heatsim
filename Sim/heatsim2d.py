import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable    
from scipy import sparse,linalg,optimize
from scipy.sparse.linalg import spsolve,splu
from Sim.utils.profiling import timer,cpuprf
from Sim.operators import DiscLaplacian,F1,dF1
from Sim.config import build_sim



def F2(cfg):
    """
    Evaluates the F2 - Pulse Reaction term given
    F2(t,x,y) = dP(t,x,y)/rho/C_p as stated in [CT, (6.3)]
    s is the "local" time in the pulse, i.e. s \belongsto [0,tpulse] 
    """
    #Sim params
    M = cfg.space.Mx
    h = cfg.space.hx
    # Material
    dEdx = cfg.target.dEdz
    rho = cfg.target.rho
    Lx = cfg.target.Lx 
    Ly = cfg.target.Ly
    C_p = cfg.target.C_p
    # beam
    sigmax = cfg.beam.sigmax
    sigmay = cfg.beam.sigmay
    Ib = cfg.beam.Ib 
    # xs = np.linspace(h,L-h,M)
    # Clearer to me
    xs = np.linspace(h,h*M,M)
    # Symmetrical mesh
    x,y = np.meshgrid(xs,xs)

    # Source code   
    # rx = L/25*(2*np.rand.uniform()-1)
    # ry = L/25*(2*np.rand.uniform()-1)
    # range [-L/25,L/25]
    ### Randomly centered at cx, cy
    # rx,ry = np.random.uniform(-L/25,L/25,2)
    
    cx = Lx/2 
    cy = Ly/2 
    dP = dEdx * Ib * rho / sigmax / sigmay/2/np.pi 
    dTdt = dP /rho / C_p
    bprofile =  np.exp(-((x-cx)**2/2/sigmax**2)-((y-cy)**2/2/sigmay**2))
    u = dTdt * bprofile
    u = np.reshape(u,(M**2,))
    return u


def ImkF1(v,k,F1,dF1,cfg) -> np.array:
    """
    Newtons' method approximation of solution to u in:
            (I - k/2*F1)u = v
    This is done to effectively approximate the inverse, to reduce compute cost. 

    """

    if hasattr(v, "toarray"):
        u = v = v.toarray()  
    else:
        u =  v

    ### TODO: Use scipy.optimize for this instead of implementing ones own, -> more flex
    for i in range(cfg.time.maxit):
        # Function and derivative values
        F1u = F1(u,cfg)
        dF1u = dF1(u,cfg)
        # Step update
        du = (u-k/2*F1u.toarray()-v) / (1-k/2 * dF1u.toarray())
        u = u - du
        # Convergence check / stopping crit
        norm_du = cfg.space.hx**(1/2)*linalg.norm(du,2)
        if norm_du < cfg.time.tol :
            break
    return u

def ImLhF1(v, L, k, F1, dF1, cfg):
    """
    Newton-Raphson method approximation of solution to u in : 
            (I - L*k/2 - k/2*F1)u = v
    This is done to approximate the inverse of nonlinear function.
    """
    M = cfg.space.Mx
    h = cfg.space.hx
    maxit = cfg.time.maxit
    tol = cfg.time.tol

    if hasattr(v, "toarray"):
        u = v = v.toarray()
    else:
        u = v

    I = sparse.eye_array(M**2, M**2)

    for i in range(maxit):
        F1u = F1(u, cfg)
        dF1u = dF1(u, cfg)

        fu = (I - L*k/2) @ u - k/2*F1u - v
        D_array = sparse.diags_array(dF1u.toarray(), offsets=0, format='csr')
        jacobian = (I - L*k/2 - k/2 * D_array)

        du = spsolve(jacobian, fu)
        u = u - du
        norm_du = h**(1/2) * linalg.norm(du, 2)
        if norm_du < tol:
            break
    return u

@cpuprf
@timer(metric="monotonic")
def modPRSolver(u0 : np.array,Laplaceh : sparse.dia_array,F1 : Callable[...,sparse.csr_array],F2 : Callable[...,sparse.csr_array],cfg):
    """
    Solves the system for u with the Modified Peaceman-Rachford scheme (TODO should reference here) 
    
    Params:
        u_0 (np.array) (M^2,) : Initial temperature array to initizialize the solves 
        Laplaceh (sparse.dia_array) : Discrete Laplacian of the system - a (M**2)x(M**2) matrix.
        F1 (function) : Non-linear reaction function
        F2 (function) : Time-dependant reaction function (Beam term)
        par (list) : Vector of all parameters.      
    Returns:

    """
    N = cfg.time.N
    M = cfg.space.Mx
    p = cfg.beam.p
    k_pause = cfg.time.k_pause
    k_pulse = cfg.time.k_pulse 
    u = np.zeros((M**2,p*2*N+1), dtype=np.float32)
    u[:,0] = u0
    Lh = Laplaceh
    I = sparse.eye_array(M**2,M**2)
    print("----- Modified Peaceman Rachford solver -----")
    print("--- Progress counter  ---")
    lu = splu((I - k_pause/2*Lh).tocsc())
    for i in range(p):
        print(f"Pulse iteration #{i+1}")
        w = np.zeros((M**2,2*N+1))
        w[:,0] = u[:,(i)*2*N]

        # Pulse step #i
        k = k_pulse
        # heating cycle
        for n in range(N):
            wn = w[:,n]
            # rhs = (I + k/2*Lh)@wn+k/2*(F2((n-1)*k,par)+F2(n*k,par))
            # w[:,n+1] = spsolve(I-k/2*Lh,rhs)
            w[:,n+1] = k*F2(cfg) + wn
            # print(k*F2(cfg))
            # wn = w[:,n]
            # rhs = (I + k/2*Lh)@wn+k/2*(F2((n-1)*k,par)+F2(n*k,par))
            # v = spsolve(I-k/2*Lh,rhs)
            # # res = ImkF1(v,par)
            # w[:,n+1] = res.reshape(M**2,)
                        

        # Pause step #i
        k = k_pause
        # lu = splu((I - k/2*Lh).tocsc())
        for n in range(N,2*N):
            ### Convert to csr to improve flops/sec
            wn = w[:,n]
            # v = (I+k/2*Lh)@(spsolve((I-k/2 * Lh),wn +k/2*F1(wn,par)))
            v = (I+k/2*Lh)@(lu.solve(wn +k/2*F1(wn,cfg)))

            w[:,n+1] = ImkF1(v,k,F1,dF1,cfg).reshape(M**2,)

        print(f"Max Temperature is under pulse is : {np.max(w)}")
        u[:,((i)*2*N+1):((i+1)*2*N+1)] = w[:,1:2*N+1]

    return u
@cpuprf
@timer(metric="monotonic")
def TrapBasic(u0, Laplaceh, F1, F2, cfg):
    """
    Solves the system without splitting, using a simple Trapezoidal time stepping scheme

    Params:
        u_0 (np.array) (M^2,) : Initial temperature array to initizialize the solves 
        Laplaceh (sparse.dia_array) : Discrete Laplacian of the system - a (M**2)x(M**2) matrix.
        F1 (function) : Non-linear reaction function
        F2 (function) : Time-dependant reaction function (Beam term)
        cfg (SimulationSetup) : Simulation configuration.
    Returns:

    """
    N = cfg.time.N
    M = cfg.space.Mx
    p = cfg.beam.p
    k_pause = cfg.time.k_pause
    k_pulse = cfg.time.k_pulse

    u = np.zeros((M**2, p*2*N+1), dtype=np.float32)
    u[:, 0] = u0
    Lh = Laplaceh
    I = sparse.eye_array(M**2, M**2)

    print("----- Trapezoidal Basic Solver -----")
    print("--- Progress counter  ---")
    for i in range(p):
        print(f"Pulse iteration #{i+1}")

        w = np.zeros((M**2, 2*N+1))
        w[:, 0] = u[:, i*2*N]
        # Pulse step #i
        k = k_pulse
        for n in range(N):
            wn = w[:, n]
            w[:, n+1] = k*F2(cfg) + wn

        # Pause step #i
        k = k_pause
        for n in range(N, 2*N):
            wn = w[:, n]
            rhs = (I + k/2*Lh) @ wn + k/2 * F1(wn, cfg)
            w[:, n+1] = ImLhF1(rhs,Lh,k, F1,dF1, cfg)

        print(f"Max Temperature is under pulse is : {np.max(w)}")
        u[:, (i*2*N+1):((i+1)*2*N+1)] = w[:, 1:2*N+1]
    return u

plot_option = 1
if __name__ == "__main__":
    
    cfg = build_sim(N = 50, Mx = 50, Lz = 0.1, p  = 10, p_freq=14)
    M = cfg.space.Mx
    N = cfg.time.N
    ### Printing parameters for simulation
    print("---------- Parameters for Simulation -------------\n",
        f"Pulse period : {1/cfg.beam.p_freq} \n",
        f"Number of pulses : {cfg.beam.p} \n", 
        f"Pulse frequency : {cfg.beam.p_freq:.2f} \n",
        f"Pulse length : {cfg.beam.t_pulse} \n",
        f"Energy : {cfg.beam.Ep} \n",
        f"Time without pulse (Pause length): {cfg.time.t_pause}\n",
        f"--- Beam and Power --- \n",
        f"Density rho : {cfg.target.rho}\n",
        f"Specific Heat Cp: {cfg.target.C_p}\n",
        f"Stopping power dEdz: {cfg.target.dEdz:.2E}\n",
        f"Peak current Ib : {cfg.beam.Ib}\n",
        f"Sigmax : {cfg.beam.sigmax} \n",
        f"Sigmay : {cfg.beam.sigmay} \n",
        f"T0 : {cfg.T0}\n")


    dP = cfg.target.rho * cfg.beam.Ib * cfg.target.dEdz
    print("dP : ",dP)
  
    # Creating Linear operators I and discrete Laplaceian Lh
    Lh = DiscLaplacian(cfg)
    
    # Initial value u_0
    u0 = np.zeros(M**2)
    # Running modified PR and IE solver to compare
    print(f"CFL Number for simulation: CFL = {cfg.time.k_pause/(cfg.space.hx**2)}")
    u = TrapBasic(u0,Lh,F1,F2,cfg,cpuprf = True)    
    u2 = modPRSolver(u0,Lh,F1,F2,cfg, cpuprf = True)
    ####   
    ##### Temporary plot to see results ####

    # Arbitrarily chosen temperature vector  
    uN = u[:,N+1]
    u2N = u2[:,N+1]

    # Absoltute and rel error
    err2Norm = linalg.norm(uN - u2N, 2) * cfg.space.hx 
    print(f"Timestep are kpulse:{cfg.time.k_pulse} and kpause = {cfg.time.k_pause}")
    print(f"Error at Time step N: {err2Norm}")
    relerr = err2Norm / np.linalg.norm(uN)
    print(f"Relative L2 Error: {relerr:.4e}")

    xs = np.linspace(0,cfg.space.Lx,M+2)
    x,y = np.meshgrid(xs,xs)

    mat = np.zeros((M+2,M+2), dtype = np.float32)
    mat[1:M+1,1:M+1] = u2[:,-1].reshape(M,M)

    if plot_option:
        u_yt = u2.reshape((M,M,-1)) [ M//2 ,: , :] # will have shape [M,N]
        plt.imshow(u_yt, cmap = 'jet',interpolation = 'nearest')
        plt.colorbar(label='Temperature')
        plt.xlabel("Time axis")
        plt.ylabel("Y-dimension")
        plt.title("Heat development over time")
        plt.show()    
        # fig,ax = plt.subplots()
        # ax.set_title(rf'Heat distribution $\delta$ T')
        plt.figure(figsize=(8,6))
        plt.imshow(mat,cmap = 'jet',interpolation= 'nearest')
        # plt.title('Heat distribution of Target material')
        plt.colorbar()
        plt.show()
   