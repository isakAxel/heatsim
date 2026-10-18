"""
    Template for 3D solver WIP

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse, linalg, special
from scipy.sparse.linalg import spsolve, splu,cg, gmres

from Sim.config import build_sim
from Sim.operators import DiscLaplacian1D, F1, dF1
from Sim.utils.profiling import timer,cpuprf
from Sim.solvers import thomasMethod
# Numerically stable sigmoid
expit = special.expit

 
from pypardiso import PyPardisoSolver

def _P_bragg():
    return 1


def F2_3D(cfg, r0=3.4):
    """
    Beam heating term dT/dt on the (Mz+2) x Mx grid (slice at y = cy), flattened
    with index = iz*Mx + ix.
    """
    Mx,My, Mz = cfg.space.Mx,cfg.space.My,cfg.space.Mz
    hx = cfg.space.hx
    Lx, Lz, Ly = cfg.target.Lx, cfg.target.Lz, cfg.target.Ly
    C_p = cfg.target.C_p
    dEdz = cfg.target.dEdz
    Ib, sx, sy = cfg.beam.Ib, cfg.beam.sigmax, cfg.beam.sigmay

    xs = np.linspace(hx, hx * Mx, Mx)          # interior points in x
    ys = xs.copy()
    zs = np.linspace(0, Lz, Mz + 2)            # includes both z boundaries
    
    x,y = np.meshgrid(xs, ys)

    cx = Lx / 2
    cy = Ly / 2

    braggc = _P_bragg(z, dEdz, r0)
    dP = Ib / C_p / sx / 2 / np.pi / sy * np.exp(-((x - cx)**2) / 2 / sx**2) * braggc
    return dP.reshape(-1)


def IkFLzThomas():
    return 1

@timer()
@cpuprf
def PRCoolingCycle2(v0, Lx,Ly, Lzs, fq, cfg,history = 0):
    """
    Dimensional splitting  Lx  /  (Lz + F1_q):

        u_n+1 = (I - k/2 (Lz+F))^-1 (I + k/2 Lx)(I - k/2 Lx)^-1 (I + k/2 (Lz+F)) u_n
    """
    N = cfg.time.N
    k = cfg.time.k_pause
    Mx,Mz = cfg.space.Mx, cfg.space.Mz
        
    
    Lx = sparse.kron(sparse.eye_array(Mz+2, Mz+2),Lx, format  = "csr") 
    Lzb = sparse.kron(Lzs,sparse.eye_array(Mx, Mx),format = "csr")
    M_l = Lx.shape[0]
    wn = v0.copy()
    I = sparse.eye_array(M_l, M_l, format="csr")
    idx = fq.nonzero()[0]
    if history:
        w = np.zeros((M_l, N + 1))
        w[:, 0] = v0.copy()

    print("------ Cooling Cycle Dimensional Split --------")
    lux = splu((I - k/2 * Lx).tocsc(), permc_spec="NATURAL")
    for n in range(N):
        
        # (I + k/2 Lz + k/2 F) w_n  -- both terms evaluated on w_n
        
        wnew = wn + k/2 * (Lzb @ wn)
        wnew[idx] += k/2 * F1(wn, cfg, fq).data
        # (I + k/2 Lx)(I - k/2 Lx)^-1
        v = (I + k/2 * Lx) @ lux.solve(wnew)
        # (I - k/2 Lz - k/2 F)^-1

        Ik
        if history:
            w[:, n + 1] = wn
    return w if history else wn