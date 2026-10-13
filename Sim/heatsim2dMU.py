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

import os, numba



# ----------------------------------------------------------------------
# Boundary conditions
# ----------------------------------------------------------------------
def BoundaryConditionNeumann(Lh, q0, qend):
    """
    Extends an (M x M) 1D Laplacian to (M+2 x M+2) by adding the two boundary
    points with ghost-point Neumann conditions.

    Returns:
        Lh (csr) : (M+2 x M+2) operator
        fq (csr) : (M+2 x 1) column marking boundary points (q0 at z=0, qend at z=Lz).
                   Its non-zero pattern is where the cooling term F1 is applied.
    """
    M = Lh.shape[0]
    _u = Lh.diagonal()[0] / -2          # = C/h^2

    top = sparse.lil_array((1, M));    top[0, 0] = 2
    bot = sparse.lil_array((1, M));    bot[0, -1] = 2
    left = sparse.lil_array((M + 2, 1));  left[0, 0] = -2;  left[1, 0] = 1
    right = sparse.lil_array((M + 2, 1)); right[-1, 0] = -2; right[-2, 0] = 1

    Lh = sparse.vstack([top * _u, Lh.tolil(), bot * _u])
    Lh = sparse.hstack([left * _u, Lh, right * _u])

    f_q = np.zeros((M + 2, 1))
    f_q[0, 0] = q0
    f_q[-1, 0] = qend
    return Lh.tocsr(), sparse.csr_array(f_q)


# ----------------------------------------------------------------------
# Beam source in the x-z plane
# ----------------------------------------------------------------------
def P_bragg(z, dEdz_0, r0, h_cut=None, peak=3.5):
    """
    Depth-dose profile of the proton beam.

    z       : depth points [cm]
    dEdz_0  : stopping power at z = 0
    r0      : Bragg peak location [cm]
    peak    : peak-to-entrance ratio
    """
    sigma_z = 0.018 * r0
    w = 0.15 * sigma_z if h_cut is None else h_cut

    peak_h = 1.0 + (peak - 1.0) * np.exp(-((z - r0)**2) / (2 * sigma_z**2))
    # peak_h = 1.0 + (peak - 1.0) * expit(-((z - r0)**2) / (2 * sigma_z**2))

    cutoff = expit(-(z - r0) / w)
    return dEdz_0 * peak_h * cutoff


def F2_xz(cfg, r0=3.4):
    """
    Beam heating term dT/dt on the (Mz+2) x Mx grid (slice at y = cy), flattened
    with index = iz*Mx + ix.
    """
    Mx, Mz = cfg.space.Mx, cfg.space.Mz
    hx = cfg.space.hx
    Lx, Lz = cfg.target.Lx, cfg.target.Lz
    C_p = cfg.target.C_p
    dEdz = cfg.target.dEdz
    Ib, sx, sy = cfg.beam.Ib, cfg.beam.sigmax, cfg.beam.sigmay

    xs = np.linspace(hx, hx * Mx, Mx)          # interior points in x
    zs = np.linspace(0, Lz, Mz + 2)            # includes both z boundaries
    x, z = np.meshgrid(xs, zs)

    cx = Lx / 2
    braggc = P_bragg(z, dEdz, r0)
    dP = Ib / C_p / sx / 2 / np.pi / sy * np.exp(-((x - cx)**2) / 2 / sx**2) * braggc
    return dP.reshape(-1)


# ----------------------------------------------------------------------
# Newton solvers for the nonlinear (implicit) steps
# ----------------------------------------------------------------------
def IkF(v, k, fq, cfg):
    """
    Newton approximation of u in  (I - k/2 F1_q) u = v
    F1_q only acts on the boundary points given by fq's pattern.
    """
    w = (cfg.space.hx * cfg.space.hz ) ** (1/2)
    idx = fq.nonzero()[0]
    u = v.copy()
    for i in range(cfg.time.maxit):
        Fu = F1(u, cfg, fq).data
        dFu = dF1(u, cfg, fq).data
        du = np.zeros_like(u)
        du[idx] = (u[idx] - k/2 * Fu - v[idx]) / (1.0 - k/2 * dFu)
        u -= du
        if w * linalg.norm(du, 2) < cfg.time.tol:
            break
    return u


def IkFLz(v, k, Lz, fq, cfg):
    """
    Newton approximation of u in  (I - k/2 Lz - k/2 F1_q) u = v


    Parameters: 
        v (array (M+2)):
            the input array for temperature distribution v
        k (float) :
            Time-step size k
        Lz (sparray (M(M+2) x M(M+2)):
            (Block-) Tridiagonal Laplacian matrix
        fq (sparray (M+2 * M,)) : 
            Boundary condition reaction term sparse vector
        cfg(SimulationSetup) : 
            Configuration object

    """
    size = Lz.shape[0]
    idx = fq.nonzero()[0]
    I = sparse.eye_array(size, size, format="csr")
    A = (I - k/2 * Lz).tocsr()                  # constant part, built once
    u = v.copy()
    # debugging
    hx, hz = cfg.space.hx, cfg.space.hz
    w = np.sqrt(hx * hz)
    temp = 1
    for i in range(cfg.time.maxit):
        Fu = F1(u, cfg, fq).data
        dFu = dF1(u, cfg, fq).data

        fu =  A @ u - v
        fu[idx] -= k/2 * Fu

        Dia = sparse.coo_array((dFu, (idx, idx)), shape=(size, size)).tocsr()
        jacobian = A - k/2 * Dia
        
        # if temp:
        #     np.savetxt("tempFu.log",Fu)        
        #     np.savetxt("tempdFu.log",dFu)
        #     temp  = 0

        du,info = cg(jacobian.tocsc(),fu,rtol = 1e-8)
        if info > 0 :
           print("not converged")
        # du = spsolve(jacobian.tocsc(), fu)

        u = u - du
        normdu = w*linalg.norm(du, 2)
        if normdu < cfg.time.tol * (1.0 + w*linalg.norm(u,2)):
            break
    if (i > 99):
        print("Iterations:", i)
    return u

def IkFLz_fast(v, k, Lhz, A1, fq, cfg):

    # A1 = (I - k/2 Lhz) as sparse CSR, built once outside the time loop
    
    Mz2, Mx = cfg.space.Mz + 2, cfg.space.Mx
    idx = fq.nonzero()[0]
    w = np.sqrt(cfg.space.hx * cfg.space.hz)

    lower = -k/2 * Lhz.diagonal(-1)
    upper = -k/2 * Lhz.diagonal(1)
    base  = 1.0 - k/2 * Lhz.diagonal(0)

    u = v.copy()
    for i in range(cfg.time.maxit):
        Fu  = F1(u, cfg, fq).data
        dFu = dF1(u, cfg, fq).data

        r = (A1 @ u.reshape(Mz2, Mx)).reshape(-1) - v
        r[idx] -= k/2 * Fu
        R = np.asfortranarray(r.reshape(Mz2, Mx))

        D = np.tile(base[:, None], (1, Mx))
        D.reshape(-1)[idx] -= k/2 * dFu      # boundary rows only
        D = np.asfortranarray(D)

        du = thomasMethod(lower, D, upper, R).reshape(-1)
        u -= du

        if w * np.linalg.norm(du) < cfg.time.tol * (1.0 + w * np.linalg.norm(u)):
            break
        if (i > 99):
            print("Iterations:", i)
    return u


# ----------------------------------------------------------------------
# Cooling cycles
# ----------------------------------------------------------------------
@timer()
def PRCoolingCycle(v0, Lx,Lz, fq, cfg, history = 0):
    """
    Splitting  Lh (full Laplacian)  /  F1_q (boundary cooling):

        u_n+1 = (I - k/2 F)^-1 (I + k/2 Lh)(I - k/2 Lh)^-1 (I + k/2 F) u_n
    """
    N = cfg.time.N
    k = cfg.time.k_pause
    Mx,Mz = cfg.space.Mx, cfg.space.Mz
    
    Ix = sparse.eye_array(Mx, Mx)
    Iz = sparse.eye_array(Mz + 2, Mz + 2)
    L = sparse.kron(Iz,Lx, format  = "csr") + sparse.kron(Lz,Ix,format = "csr")
    I = sparse.kron(Ix,Iz,format = "csr")
    # I = sparse.eye_array(M_l, M_l, format="csr")
    M_l = L.shape[0]

    wn = v0.copy()
    
    idx = fq.nonzero()[0]
    if history:
        w = np.zeros((M_l,N+1))
        w[:,0] = v0

    print("------ Cooling Cycle Boundary Term Split --------")
    ## TODO: Change to paralell solver
    lu = splu((I - k/2 * L).tocsc(), permc_spec="NATURAL")
    
    for n in range(N):
        wn[idx] += k/2 * F1(wn, cfg, fq).data           # (I + k/2 F) w
        v = (I + k/2 * L) @ solver.solve()
        wn = IkF(v, k, fq, cfg)
        if history:
            w[:,n+1] = wn
    return w if history else wn

fast = 1
@timer()
def PRCoolingCycle2(v0, Lx, Lzs, fq, cfg,history = 0):
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

        if fast == 1:
            # if n == 0:
            #     print("Using Thomas!")
            
            wn = IkFLz_fast(v,k,Lzs,(sparse.eye_array(Mz+2, Mz+2) - k/2 * Lzs),fq,cfg)
        else:
            wn = IkFLz(v, k, Lzb, fq, cfg)
        if history:
            w[:, n + 1] = wn
    return w if history else wn

@timer()
def HeunCoolingCyclev2(v0,Lx,Lz, fq, cfg, history = 0):
    """
    Explicit trapezoidal rule (Heun's method) with NO splitting, applied to
        u' = L u + F1_q(u)
 
        k1 = L u_n + F(u_n)
        k2 = L (u_n + k k1) + F(u_n + k k1)
        u_n+1 = u_n + k/2 (k1 + k2)
 
    Second order in k. Fully explicit, so k is limited by the stability bound
    of L (roughly k*C*(1/hx^2 + 1/hz^2) <~ 0.5 for Heun, boundary rows stricter).
 
    Parameters:
        v0 (array): initial value, flattened (index = iz*Mx + ix)
        L (sparse): full operator Lx + Lz
        fq (sparse column): boundary marker, non-zero where F1 is applied
        cfg (SimulationSetup)
    """
    N = cfg.time.N
    k = cfg.time.k_pause
    L = (Lx + Lz).tocsr()
    M_l = L.shape[0]
   
    wn = v0.copy()
    idx = fq.nonzero()[0]
    if history:
        w = np.zeros((M_l, N + 1))
        w[:, 0] = v0.copy()

    def rhs(u):
        r = L @ u
        r[idx] += F1(u, cfg, fq).data
        return r
 
    print("------ Cooling Cycle Explicit Trapezoidal (Heun), no splitting --------")
    for n in range(N):
        u = wn
        k1 = rhs(u)
        k2 = rhs(u + k * k1)
        wn = u + k / 2 * (k1 + k2)
        if history:
            w[:,n+1] = wn
    return w if history else wn
 
 
@timer()
def TrapCoolingCycle(v0,Lx,Lz, fq, cfg, history = 0):
    """
    Implicit trapezoidal rule with NO splitting (second order in k). Used as the
    reference for the splitting errors.
 
        (I - k/2 L - k/2 F) u_n+1 = (I + k/2 L) u_n + k/2 F(u_n)
 
    Parameters:
        v0 (array): initial value, flattened
        L (sparse): full operator Lx + Lz
        fq (sparse column): boundary marker
        cfg (SimulationSetup)
    """
    N = cfg.time.N
    k = cfg.time.k_pause
    L = (Lx + Lz).tocsr()
    M_l = L.shape[0]
 
    wn = v0.copy()
    
    I = sparse.eye_array(M_l, M_l, format="csr")
    B = (I + k/2 * L).tocsr()                       # explicit part, built once
    idx = fq.nonzero()[0]
    
    if history: 
        w = np.zeros((M_l, N + 1))
        w[:, 0] = v0.copy()

    print("------ Cooling Cycle Implicit Trapezoidal Rule Solver --------")
    for n in range(N):
        wold = wn
        rhs = B @ wold
        rhs[idx] += k/2 * F1(wold, cfg, fq).data
        wn = IkFLz(rhs, k, L, fq, cfg)
        if history:
            w[:,n+1] = wn
    
    return w if history else wn


# ----------------------------------------------------------------------
# Plot helper
# ----------------------------------------------------------------------
def plot_field(cfg, u_vec, title, method, mark_max=False,save_fig=False):
    Mx, Mz = cfg.space.Mx, cfg.space.Mz
    hx = cfg.space.hx
    N = cfg.time.N
    x = np.linspace(hx, hx * Mx, Mx)
    z = np.linspace(0, cfg.target.Lz, Mz + 2)

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(x, z, u_vec.reshape((Mz + 2, Mx)), cmap="jet", shading="auto")
    plt.colorbar(mesh, label="Temperature")
    ax.set_xlabel("x-axis [cm]")
    ax.set_ylabel("z-axis [cm]")
    ax.set_title(title + f"Method:{method}")
    if mark_max:
        iz, ix = np.unravel_index(np.argmax(u_vec), (Mz + 2, Mx))
        ax.plot(x[ix], z[iz], marker="D", markersize=6, markerfacecolor="white",
                markeredgecolor="black", label=f"Max Temperature = {u_vec[iz*Mx+ix]:.1f}")
        ax.legend(loc="upper right")
    if save_fig:
        path = f"./benchmarks/data/imgs/"
        fig.savefig(path + f"Temp_{method}_M{Mx}_N{N}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)     
    else:
        plt.show()


def plot_compare(cfg, ua, ub, labels=("Dimensional split", "Boundary split")):
    """
    Compare two final temperature distributions, each of shape ((Mz+2)*Mx,).
    (If a history array of shape ((Mz+2)*Mx, N+1) is passed, its last column is used.)

    Top row    : field A, field B, and their difference (A - B)
    Bottom row : z-profile through the beam center, x-profile at the depth of
                 max temperature of A, and the profile difference along z
    """
    Mx, Mz = cfg.space.Mx, cfg.space.Mz
    hx = cfg.space.hx
    x = np.linspace(hx, hx * Mx, Mx)
    z = np.linspace(0, cfg.target.Lz, Mz + 2)

    ua, ub = np.asarray(ua), np.asarray(ub)
    if ua.ndim == 2:
        ua = ua[:, -1]
    if ub.ndim == 2:
        ub = ub[:, -1]

    fa = ua.reshape((Mz + 2, Mx))
    fb = ub.reshape((Mz + 2, Mx))
    diff = fa - fb
    vmin, vmax = min(fa.min(), fb.min()), max(fa.max(), fb.max())

    ix_c = Mx // 2
    iz_m, _ = np.unravel_index(np.argmax(fa), fa.shape)

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    for a, f, lab in ((ax[0, 0], fa, labels[0]), (ax[0, 1], fb, labels[1])):
        m = a.pcolormesh(x, z, f, cmap="jet", shading="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(m, ax=a, label="Temperature")
        a.set_title(lab)
        a.set_xlabel("x [cm]"); a.set_ylabel("z [cm]")

    m = ax[0, 2].pcolormesh(x, z, diff, cmap="coolwarm", shading="auto")
    plt.colorbar(m, ax=ax[0, 2], label="Temperature difference")
    ax[0, 2].set_title(f"Difference ({labels[0]} - {labels[1]})")
    ax[0, 2].set_xlabel("x [cm]"); ax[0, 2].set_ylabel("z [cm]")

    # z-profile at beam center
    ax[1, 0].plot(z, fa[:, ix_c], "-", label=labels[0])
    ax[1, 0].plot(z, fb[:, ix_c], "--", label=labels[1])
    ax[1, 0].set_title(f"z-profile at x = {x[ix_c]:.2f} cm")
    ax[1, 0].set_xlabel("z [cm]"); ax[1, 0].set_ylabel("Temperature")
    ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

    # x-profile at depth of max temperature
    ax[1, 1].plot(x, fa[iz_m, :], "-", label=labels[0])
    ax[1, 1].plot(x, fb[iz_m, :], "--", label=labels[1])
    ax[1, 1].set_title(f"x-profile at z = {z[iz_m]:.2f} cm")
    ax[1, 1].set_xlabel("x [cm]"); ax[1, 1].set_ylabel("Temperature")
    ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)

    # difference along z at beam center
    ax[1, 2].plot(z, diff[:, ix_c], "k-")
    ax[1, 2].axhline(0, color="gray", lw=0.8)
    ax[1, 2].set_title(f"Difference along z at x = {x[ix_c]:.2f} cm")
    ax[1, 2].set_xlabel("z [cm]"); ax[1, 2].set_ylabel("Temperature difference")
    ax[1, 2].grid(alpha=0.3)

    print(f"Max |difference| in final field: {np.abs(diff).max():.4e}")
    print(f"Max temperature  {labels[0]}: {fa.max():.4f}   {labels[1]}: {fb.max():.4f}")

    fig.tight_layout()
    plt.show()
 
 
# ----------------------------------------------------------------------
if __name__ == "__main__":
    plot_option = 1
    M = 600
    cfg = build_sim(N=600, Mx=M, Mz=M, t_pulse=1.5e-3)
    Mx, Mz, N = cfg.space.Mx, cfg.space.Mz, cfg.time.N
    h = cfg.space.hx

    numba.set_num_threads(1)

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
    Lz_s = DiscLaplacian1D(cfg, axis="z")
    Lz_s, fq1d = BoundaryConditionNeumann(Lz_s, q0=1, qend=0)

    # print("L_z matrix :"  )
    # np.savetxt("testmatrixLz",Lz_s.toarray(),fmt = "%.2f")
    # exit()
    # Lx = sparse.kron(Iz, Lhx, format="csr")
    # Lz = sparse.kron(Lhz, Ix, format="csr")
    fq = sparse.kron(fq1d, np.ones((Mx, 1)), format="csr")

    # --- Initial value after heating pulse ----------------------------
    v0 = F2_xz(cfg) * cfg.beam.t_pulse
    # print("Temperature max after pulse:", np.max(v0))
    # if plot_option:
    #     plot_field(cfg, v0, "Heat after 1st heating cycle\n", "Initial Heating Cycle")


    # --- Solve --------------------------------------------------------
    print("Numba uses:", numba.get_num_threads())
    u = PRCoolingCycle2(v0, Lx_s, Lz_s, fq, cfg)
    numba.set_num_threads(2)
    print("Numba uses:", numba.get_num_threads())
    u2 = PRCoolingCycle2(v0, Lx_s, Lz_s, fq, cfg)
    numba.set_num_threads(4)
    print("Numba uses:", numba.get_num_threads())
    u3 = PRCoolingCycle2(v0, Lx_s, Lz_s, fq, cfg)
        
    # u2 = PRCoolingCycle(v0,Lx_s,Lz_s,fq,cfg)
    # fast = 0
    # u3 = PRCoolingCycle2(v0, Lx_s, Lz_s, fq, cfg)
    
    # err2Norm = linalg.norm(u2 - u, 2) * h
    # print(f" error L2-norm : {err2Norm}")

    # # --- Diagnostics --------------------------------------------------
    # diff_coeff = cfg.target.D / (cfg.target.rho * cfg.target.C_p)
    # diff_length = np.sqrt(diff_coeff * cfg.time.t_pause)
    # print(f"Diffusion length: {diff_length:.4f} cm, grid spacing h: {h:.4f} cm")
    # print(f"That's {diff_length/h:.1f} grid cells")
    # print(f"dt = {max(cfg.time.k_pause, cfg.time.k_pulse)}")
    # print(f"CFL Number for simulation: CFL = {cfg.time.k_pause/(h**2)}")


    # if plot_option:
    #     plot_field(cfg, u,
    #                f"Heat after 1st cooling cycle,\n M = {Mx}, N = {N}. ",
    #                "Peaceman Rachford",
    #                mark_max=True)

    # plot_compare(cfg,u,u3,labels = ('Dimensional Split', 'Boundary Split'))
