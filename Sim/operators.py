import numpy as np
from scipy import sparse,fft,linalg


"""
    Spatial Discretization
"""
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
    sigma, epsilon = cfg.sigma, cfg.target.epsilon1
    gamma = cfg.target.gamma1
    
    c_bolt = sigma*epsilon/rho/C_p 
    c_newton = gamma/rho/C_p
    if fq is not None:
        hz = cfg.space.hz
        ### TODO: idx optimization precompute
        #
        idx = fq.nonzero()[0]          # entries with fq == 0 are not included
        is1 = fq.data == 1             # same order as idx; everything else is treated as 2
        gamma = np.where(is1, cfg.target.gamma1, cfg.target.gamma2)
        epsilon = np.where(is1, cfg.target.epsilon1, cfg.target.epsilon2)

        c_bolt = sigma*epsilon/rho/C_p 
        c_newton = gamma/rho/C_p

        fu = fq.copy()
        fu.data = -(c_bolt*(((v[idx]+T0)/100.0)**4-(T0/100.0)**4) + c_newton*v[idx]) * 2/hz
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
    sigma, epsilon = cfg.sigma, cfg.target.epsilon1
    gamma = cfg.target.gamma1
    # Coefficients
    c_bolt =sigma*epsilon/rho/C_p
    c_newton = gamma /rho/C_p
    if fq is not None:
        hz = cfg.space.hz
        # Try this for optimizing Later
        idx = fq.nonzero()[0]          # entries with fq == 0 are not included
        is1 = fq.data == 1             # same order as idx; everything else is treated as 2
        gamma = np.where(is1, cfg.target.gamma1, cfg.target.gamma2)
        epsilon = np.where(is1, cfg.target.epsilon1, cfg.target.epsilon2)
        c_bolt = sigma*epsilon/rho/C_p 
        c_newton = gamma/rho/C_p
        fu = fq.copy()
        fu.data = (-4*c_bolt*((v[idx]+T0)/100)**3 *1/100 - c_newton) * 2/hz 
        return fu 
    else:
        A,V = cfg.target.A, cfg.target.V
        u = -(4*c_bolt*((v + T0)/100)**3 * 1/100 + c_newton ) * A/V ### Element wise power ### extra factor 100 here
        return sparse.csr_array(u)

# ----------------------------------------------------------------------
# FFT Stuff
# ----------------------------------------------------------------------
def lambda_1DFFT(c,h):
    """
        Retrieve the eigenvalues from the first fourier node of the discrete laplacian c

        Params:
            c (ndarray)

    """
    tol = 1e-13
    c = np.asarray(c,dtype = float)
    lam = fft.fft(c)
    # Warn when not symmetric
    if linalg.norm(lam.imag, 2) > tol:
        print("Warning: Imaginary values are greater than 0")
    return lam.real / h**2

def lambda_xyFFT(cx,hx,cy,hy):
    """
        Get 2D-eigenvalues from the fourier modes cx,cy - uses the 1D eigenvalues and compiles them.
    """
    lam_x = lambda_1DFFT(cx,hx)
    lam_y = lambda_1DFFT(cy,hy)

    lam_xy = lam_x[:, None] + lam_y[None, :]  

    return lam_xy


def fft_apply(U,lamxy,D,dt):
    """
        Computes a discrete fourier transform on the temperature distribution U, to then apply the eigen values of the
        laplace "solution" operator, to then transform back to the spatial domain before returning the temperature distribution.
        
        Periodic boundary conditions are assumed as of now and the function is hard-coded for 2D slices of the ordering of U.

        Params:
            U (ndarray) : 3D - temperature grid with shapes (mx,my,mz)
            lamxy(ndarray) : 3D array of the eigenvalues
            Mx (int) : "spatial" resolution in x
            My (int) : same
            D (float) : The thermal diffusivity constant
            dt(float) : the time step to solve for (analagous to k)
    """
    U_hat = fft.fft2(U,axes = (0,1))
    U_hat *= np.exp(D * lamxy*dt)
    U = fft.ifft2(U_hat,axes = (0,1)).real
    return U
   


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

def P_bragg_cons(z,dEdz):
    """
    Constant depth-dose profile of proton beam. Recast as a ndarray in z-direction
    
    Params:
        z (ndarray) : Discretized points in z-direction (Proton beam incident direction)
        dEdz(float) : The constant stopping power of the material given in [MeV cm^2 / g]
    """
    return np.full_like(z,dEdz, dtype=float)

def F2_xz(cfg, r0=3.4):
    """
    Beam heating term dT/dt on the (Mz+2) x Mx grid (slice at y = cy), flattened
    with index = iz*Mx + ix.
    """
    Mx, Mz = cfg.space.Mx, cfg.space.Mz
    hx = cfg.space.hx
    Lx, Lz = cfg.target.Lx, cfg.target.Lz
    C_p = cfg.target.C_p
    dEdz = cfg.target.dEdz[1][0]
    Ib, sx, sy = cfg.beam.Ib, cfg.beam.sigmax, cfg.beam.sigmay

    xs = np.linspace(hx, hx * Mx, Mx)          # interior points in x
    zs = np.linspace(0, Lz, Mz + 2)            # includes both z boundaries
    x,z= np.meshgrid(xs,zs, indexing ="ij")  # Change reference definition

    cx = Lx / 2
    braggc = P_bragg(z, dEdz, r0)
    # Using GaussianBeam profile
    dP = Ib * braggc / C_p *  GaussianBeam(x,0,cx,0,sx,sy)
    return dP.reshape(-1)

def F2_xyz(cfg,beam_mode="Gaussian", r0=3.4):
    """
    Beam heating term dT/dt on the Mx x My x (Mz+2) grid 
    with index = (ix*My + iy)*(Mz+2) + iz
    """
    Mx, Mz,My = cfg.space.Mx, cfg.space.Mz, cfg.space.My
    hx,hy  = cfg.space.hx, cfg.space.hy
    Lx, Ly, Lz = cfg.target.Lx,cfg.target.Ly, cfg.target.Lz
    C_p = cfg.target.C_p
    dEdz = cfg.target.dEdz[1][0]
    Ib, sx, sy = cfg.beam.Ib, cfg.beam.sigmax, cfg.beam.sigmay

    xs = np.linspace(hx, hx * Mx, Mx)          # interior points in x,y
    ys = np.linspace(hy,hy * My, My) 
    zs = np.linspace(0, Lz, Mz + 2)            # includes both z boundaries
    x,y,z = np.meshgrid(xs,ys,zs, indexing ="ij")  # Change reference definition

    cx = Lx / 2
    cy = Ly / 2
    braggc = P_bragg(z, dEdz, r0)
    dP = braggc * Ib / C_p
    if beam_mode == "Gaussian":
        u = dP * GaussianBeam(x,y,cx,cy,sx,sy)
    # dP = Ib / C_p / sx / 2 / np.pi / sy * np.exp(-((x - cx)**2) / 2/ sx**2 - ((y-cy)**2)/2 /sy**2) * braggc
    return u.reshape(-1)

def F2_xyz_MC(cfg,beam_mode="Gaussian"):
    """
    Beam heating term dT/dt on the Mx x My x (Mz+2) grid 
    with index = (ix*My + iy)*(Mz+2) + iz
    """
    Mx, Mz,My = cfg.space.Mx, cfg.space.Mz, cfg.space.My
    hx,hy  = cfg.space.hx, cfg.space.hy
    Lx, Ly, Lz = cfg.target.Lx,cfg.target.Ly, cfg.target.Lz
    C_p = cfg.target.C_p
    Ib, sx, sy = cfg.beam.Ib, cfg.beam.sigmax, cfg.beam.sigmay

    xs = np.linspace(hx, hx * Mx, Mx)          # interior points in x,y
    ys = np.linspace(hy,hy * My, My) 
    zs = np.linspace(0, Lz, Mz + 2)            # includes both z boundaries
    x,y,z = np.meshgrid(xs,ys,zs, indexing ="ij")  # Change reference definition

    cx = Lx / 2
    cy = Ly / 2
    # Bragg peak prfile
    zsrc = cfg.target.dEdz[0] * 1e-8 
    ysrc = cfg.target.dEdz[1]
    braggc = np.interp(zs,zsrc,ysrc)
    dP = braggc * Ib / C_p
    if beam_mode == "Gaussian":
        u = dP * GaussianBeam(x,y,cx,cy,sx,sy)
    # dP = Ib / C_p / sx / 2 / np.pi / sy * np.exp(-((x - cx)**2) / 2/ sx**2 - ((y-cy)**2)/2 /sy**2) * braggc
    return u.reshape(-1)        

"""
    Beam modes specifications
"""

def GaussianBeam(x,y,cx,cy,sx,sy):
    """
        Returns a gaussian beam profile for discretized x,y-values x and y

    """
    return np.exp(-((x - cx)**2) / 2/ sx**2 - ((y-cy)**2)/2 /sy**2) / ( 2 * np.pi * sx * sy)