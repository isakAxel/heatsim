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

def DiscLaplacian1D(cfg) -> sparse.sparray[float]:
    """
    Creating discrete Laplacian Lh for 1D - a (M x M) matrix
    """
    M = cfg.space.Mx
    h = cfg.space.hx
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
        print("hello")
        idx = fq.nonzero()[0]
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
        fu.data = (-4*c_bolt*((v[idx]+T0)/100)**3 *1/100 - c_newton) * 2/hz 
        return fu 
    else:
        A,V = cfg.target.A, cfg.target.V
        u = -(4*c_bolt*((v + T0)/100)**3 * 1/100 + c_newton ) * A/V ### Element wise power ### extra factor 100 here
        return sparse.csr_array(u)