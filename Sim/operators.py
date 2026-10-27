import numpy as np
from scipy import sparse

def DiscLaplacian(cfg) -> sparse.sparray[float]:
    """
    Creating the discrete Laplaceian Lh - an (M**2)x(M**2) matrix
    """
    M = cfg.space.Mx
    h = cfg.space.hx
    rho, C_p, D = cfg.target.rho, cfg.target.C_p, cfg.target.D
    
    C = D / rho / C_p
    ve = np.ones(M)
    K = C/h**2 * sparse.dia_array(([ve,-2*ve,ve],[-1,0,1]),shape = (M,M))
    Lh = sparse.kronsum(K,K)    ### kron(K,I) + kron(I,K) 
    return Lh

def DiscLaplacian1D(cfg,axis = "x") -> sparse.sparray[float]:
    """
    Creating discrete 1D Laplacian Lh along a given axis {"x","y","z"} - a (M x M) matrix
    """
    if axis == "x":
        M = cfg.space.Mx
        h = cfg.space.hx
    elif axis == "y":
        M = cfg.space.My
        h = cfg.space.hy
    elif axis == "z":
        M = cfg.space.Mz
        h = cfg.space.hz
    else:
        raise ValueError("axis must be x y or z")
        
    rho, C_p, D = cfg.target.rho, cfg.target.C_p, cfg.target.D

    C = D/rho / C_p
    ve = np.ones(M)
    Lx = C/h**2 * sparse.dia_array(([ve,-2*ve,ve],[-1,0,1]),shape = (M,M))
    return Lx

def F1(v,cfg,fq = None) -> sparse.sparray:
    """
    Evaluates F1 -radiation term given by Stefan-Boltzmann Law and Newton Convective cooling

    Params: 
        v(np.array):
            Temperature distr array to be evaluated by F1
    
    Returns:

    """
    if hasattr(v, "toarray"):
        v = v.toarray()
    # TODO TEST hopefully both of F1 and dF1 can be used for MU and 2D

    # Get params from cfg
    T0 = float(cfg.T0)
    rho, C_p = cfg.target.rho, cfg.target.C_p
    sigma, epsilon = cfg.sigma, cfg.target.epsilon
    gamma = cfg.target.gamma
    
    c_bolt = sigma*epsilon/rho/C_p 
    c_newton = gamma/rho/C_p
    if fq is not None:
        hz = cfg.space.hz
        ### TODO: idx optimization precompute
        #
        idx = fq.nonzero()[0]
        fu = fq.copy()
        fu.data = -(c_bolt*(((v[idx]+T0)/100.0)**4-(T0/100.0)**4) + c_newton*v[idx]) * 1/hz
        return fu
    else:
        A,V = cfg.target.A, cfg.target.V
        u =  - (c_bolt*(((v+T0)/100.0)**4-(T0/100.0)**4) + c_newton*v) * A/V
        return sparse.csr_array(u)
    
def dF1(v,cfg,fq = None) -> sparse.sparray:
    """
    Evaluates the derivate wrt to v of F1(v)-radiation term

    """
    if hasattr(v, "toarray"):
        v = v.toarray()

    T0 = float(cfg.T0)
    rho, C_p = cfg.target.rho, cfg.target.C_p
    sigma, epsilon = cfg.sigma, cfg.target.epsilon
    gamma = cfg.target.gamma

    # Coefficients
    c_bolt =sigma*epsilon/rho/C_p
    c_newton = gamma /rho/C_p
    
    if fq is not None:
        hz = cfg.space.hz
        # Try this for optimizing Later
        # if idx is None:
            # idx = fq.nonzero()[0]
        idx = fq.nonzero()[0]
        fu = fq.copy()
        fu.data = (-4*c_bolt*((v[idx]+T0)/100)**3 *1/100 - c_newton) * 1/hz 
        return fu 
    else:
        A,V = cfg.target.A, cfg.target.V
        u = -(4*c_bolt*((v + T0)/100)**3 * 1/100 + c_newton ) * A/V ### Element wise power ### extra factor 100 here
        return sparse.csr_array(u)

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
    from scipy.special import expit
    
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
    x,z= np.meshgrid(xs,zs, indexing ="ij")  # Change reference definition

    cx = Lx / 2
    braggc = P_bragg(z, dEdz, r0)
    dP = Ib / C_p / sx / 2 / np.pi / sy * np.exp(-((x - cx)**2) / 2 / sx**2) * braggc
    return dP.reshape(-1)

def F2_xyz(cfg, r0=3.4):
    """
    Beam heating term dT/dt on the Mx x My x (Mz+2) grid 
    with index = (ix*My + iy)*(Mz+2) + iz
    """
    Mx, Mz,My = cfg.space.Mx, cfg.space.Mz, cfg.space.My
    hx,hy  = cfg.space.hx, cfg.space.hy
    Lx, Ly, Lz = cfg.target.Lx,cfg.target.Ly, cfg.target.Lz
    C_p = cfg.target.C_p
    dEdz = cfg.target.dEdz
    Ib, sx, sy = cfg.beam.Ib, cfg.beam.sigmax, cfg.beam.sigmay

    xs = np.linspace(hx, hx * Mx, Mx)          # interior points in x,y
    ys = np.linspace(hy,hy * My, My) 
    zs = np.linspace(0, Lz, Mz + 2)            # includes both z boundaries
    x,y,z = np.meshgrid(xs,ys,zs, indexing ="ij")  # Change reference definition

    cx = Lx / 2
    cy = Ly / 2
    braggc = P_bragg(z, dEdz, r0)
    dP = Ib / C_p / sx / 2 / np.pi / sy * np.exp(-((x - cx)**2) / 2/ sx**2 - ((y-cy)**2)/2 /sy**2) * braggc
    return dP.reshape(-1)