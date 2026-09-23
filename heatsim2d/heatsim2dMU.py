from heatsim2d import define_parameters,set_nums,DiscLaplacian1D,use_parameters,set_xspace
from utils.profiling import timer
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse,linalg
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu

MeV = 1e6
# Setting params
Ep,material,dS_,rho,C_p,p_freq,p,t_pulse,Ib,sigma_x,sigma_y,epsilon,D,L,T0,gamma,plot_option = define_parameters(t_pulse = 0.0015)
k_pulse,k_pause,maxit,tol,N = set_nums()

h,M = set_xspace(L,Mx=200)

tend = p / p_freq
t_pause = (tend - p*t_pulse) / p
sigma  = 5.67037442e-4               # 1e8 * Stefan-Boltzmann constant (W/((cm)^2 K^4)
A = L**2
t = 0.1                             # cm
V = A * t
dEdz = dS_ * MeV                    # Blades-Nickel (eV cm^2/g)

par = [M,h,N,k_pulse,maxit,tol,T0,L,tend,rho,C_p,D,sigma,epsilon,A,V,dEdz,Ib,sigma_x,sigma_y,gamma];

def BoundaryConditionNeumann(Lh,h,q0,qend):
    """
    Adds boundary condition row to simple Discrete Laplacian
        (assumes Lh to be a sparse matrix)
    """
    # Lh = sparse.dia_array()
    
    

    M,_ = Lh.shape
    Lh = Lh.tolil()
    _u = Lh[0,0] / -2
    # _u = 1 
    ### adding rows
    top = sparse.lil_array((1,M))
    bot = sparse.lil_array((1,M))
    top[0,0] = 2 
    bot[0,-1] = 2

    # adding cols
    left = sparse.lil_array((M+2,1))
    right = sparse.lil_array((M+2,1))
    # Same h right now
    left[0,0] = -2
    left[1,0] = 1
    right[-1,0] = -2 
    right[-2,0] = 1 
    Lh = sparse.vstack([top*_u,Lh,bot*_u])
    Lh = sparse.hstack([left*_u,Lh,right*_u])

    # Constructing boundary conditions neumann
    # input q0 = 1 to later use function

    f_q = np.zeros((M,1))
    f_q = np.vstack(([q0],f_q,[qend]), dtype = float)
    f_q = sparse.csr_array(f_q)

    # Checking shape
    # print(Lh.toarray(),f_q.toarray())

    return Lh.tocsr(),f_q

### Right now it is a general matrix but probablt just use 1D vector ?
@use_parameters(1,6,9,10,12,13,20)
def Fq(v,fq,h,T0,rho,C_p,sigma,epsilon,gamma):
    """
    Evaluates the boundary term  i.e. cooling terms given by Stefan-Boltzmann Law and Newton's law of cooling,
    fq evaluates the cooling effects on the boundary as a neumann condition.
    """
    u = fq.copy()
    if sparse.issparse(v):
        v_vals = fq.multiply(v).data
    else:
        # If v is a dense 1D or 2D array, sample it using Fq's non-zero indices
        v_vals = v[fq.nonzero()[0]]

    T0 = float(T0)
    c_bolt = sigma*epsilon/rho/C_p 
    c_newton = gamma/rho/C_p
    cdata = - (c_bolt*(((v_vals+T0)/100.0)**4-(T0/100.0)**4)) - c_newton*v_vals
    u.data = cdata * h/2
    return u

@use_parameters(1,6,9,10,12,13,20)
def dFq(v,fq,h,T0,rho,C_p,sigma,epsilon,gamma):
    """
    Evaluates the derivate wrt v of Fq(v) the boundary reaction term, constructed to find solution with Newtonsolver
    """    
    u = fq.copy()
    if sparse.issparse(v):
        v_vals = fq.multiply(v).data
    else:
        v_vals = v[fq.nonzero()[0]]

    T0 = float(T0)
    c_bolt = sigma*epsilon/rho/C_p
    c_newton = gamma/rho/C_p
    #TODO Mabe negative ? 
    cdata = - 4*c_bolt*((v_vals+T0)/100)**3 *1/100 - c_newton 
    u.data = cdata *h/2
    return u

@use_parameters(1,3,4,5,6,9,10,12,13,20)
def IkFq(v,fq,h,k,maxit,tol,T0,rho,C_p,sigma,epsilon,gamma):
    """ 
    Newton's method approximatioj of sol to u in:
        (I -k/2 * Fq) u = v
    This is done to effectively approximate the inverse.

    """
    if hasattr(v, "toarray"):
        u = v = v.toarray()  
    else:
        u =  v
    for i in range(maxit):
        # F and F_prime
        Fqu = Fq.__wrapped__(u,fq,h,T0,rho,C_p,sigma,epsilon,gamma)
        dFqu = dFq.__wrapped__(u,fq,h,T0,rho,C_p,sigma,epsilon,gamma)
        # Step update
        idx = Fqu.indices
        fdata = Fqu.data
        dFdata = dFqu.data
        # Sparse implementation to only update nonzero indices
        du = np.zeros_like(u)
        du[idx] = (u[idx] - (k / 2.0) * fdata - v[idx]) / (1.0 - (k / 2.0) * dFdata) 

        # Stepupdate and check stoppingcrit
        u = u - du
        norm_du = h**(1/2)*linalg.norm(du,2)
        if norm_du < tol :
            if(i>1):
                print(f"iterations i : {i}")    
            break 
    
    return u
@use_parameters(0,1,7,9,10,16,17,18,19)
def F2_t(s,M,h,L,rho,C_p,dEdz,Ib,sigma_x,sigma_y):
    """

    """

    # [h,L-h]
    xs = np.linspace(h,h*M,M)
    zs = np.linspace(0,L,M+2)
    x,z = np.meshgrid(xs,zs)
    cx = L/2
    r0 = 3.4
    # dEdz_v = np.ones(M) * dEdz
    dP = Ib/C_p/sigma_x/2/np.pi/sigma_y*np.exp(-((x-cx)**2/2/sigma_x**2)) * P_bragg(z,dEdz,r0)
    u = dP.reshape(-1,)
    return u

def P_bragg(z,dEdz_0,r0,peak = 3.5):
    """
        Computes depth-dose distribution of incidnent proton beam.

        Parameters:
            z:
                Array of spatial depth points [cm]
            dEdz_0:
                Constant stopping power at given energy at z = 0 [MeV/cm]
            r0:
                Bragg peak location  [cm]
            peak:
                Peak-to-entrace ratio (defaults as 3.5)
        Returns:
            dEdz : Stopping power profile in [MeV/cm]

    """
    # Peak parameters
    sigma_z = 0.018*r0
    w = 0.15*sigma_z

    peak_h = 1.0 + (peak - 1.0) * np.exp(-((z - r0)**2) / (2 * sigma_z**2))
    cutoff = 1.0 / (1.0 + np.exp((z - r0) / w))
    P_tilde = peak_h * cutoff


    # P = (D_entrance + peak) * cutoff
    return dEdz_0*P_tilde

@timer()
def PRCoolingCycle(v0,Lh,fq,par):

    M_l = Lh.shape[0]
    N = par[2]

    w = np.zeros((M_l,N+1))
    w[:,0] = v0
    I = sparse.eye_array(M_l,M_l)

    print("------ Cooling Cycle --------")
    par[3] = k = k_pause
    lu = splu(I - k/2*Lh, permc_spec = 'NATURAL')
    for n in range(N):
        wn = w[:,n].copy()
        ## Sparse add
        fqu = Fq(wn, fq, par)          # fqu is sparse csr vector 
        idx = fqu.indices
        wn[idx] += k/2* fqu.data
        v = (I +k/2*Lh) @ lu.solve(wn)
        w[:,n+1] = IkFq(v,fq,par)

    return w


if __name__ == "__main__":
    
    ### Printing parameters for simulation
    print( "---------- Parameters for Simulation -------------\n",
        f"Pulse period : {1/p_freq} \n",
        f"Pulse frequency : {p_freq:.2f} \n",
        f"Pulse length : {t_pulse} \n",
        f"Energy : {Ep} \n",
        f"Time without pulse (Pause length): {t_pause}\n",
        f"--- Beam and Power --- \n",
        f"Density rho : {rho}\n",
        f"Specific Heat Cp: {C_p}\n",
        f"Stopping power dEdz: {dEdz:.2E}\n",
        f"Peak current Ib : {Ib}\n",
        f"Sigmax : {sigma_x} \n",
        f"Sigmay : {sigma_y} \n",
        f"T0 : {T0}\n",)

    Lhx = DiscLaplacian1D(0,par)
    Lhz = DiscLaplacian1D(0,par)
    # print(Lh.toarray())
    Lhz, fq = BoundaryConditionNeumann(Lhz,h,q0=1,qend=0)
    
    
    one_x = np.ones((M,1))
    one_z = np.ones((M+2,1))
    # Create t
    fq = sparse.kron(fq,one_x, format='csr')
    Ix = sparse.eye_array(M,M)
    Iz = sparse.eye_array((M+2),(M+2))
    # L_2D = sparse.kron(Lhz,Ix, format='csr') + sparse.kron(Iz,Lhx, format='csr')
    L_2D = sparse.kronsum(Lhx,Lhz, format = 'csr')
    # print(Fq_2D.shape)
    # print(type(Fq_2D))
    # print(L_2D.shape)
    
    # v0 shape ((M+2)M,1)
    v0 = F2_t(0,par) * t_pulse
    plot = plot_option
    if plot:
        plt.imshow(v0.reshape((M+2,M)), cmap = 'jet',interpolation = 'nearest')
        plt.colorbar(label='Temperature')
        plt.xlabel("x-axis")
        plt.ylabel("z-axis")
        plt.title("Heat after 1st heating cycle")
        plt.show()    


    # plt.spy(L_2D)
    # plt.show()

    tb = 0
    if tb:
        z = np.linspace(0,L,20000)
        r0 = 3.4
        print(f"dEdz = {dEdz}")
        braggc = P_bragg(z,dEdz,r0)
        plt.plot(z,braggc,'bx')
        plt.show()

    u = PRCoolingCycle(v0,L_2D,fq,par)

    if plot_option:
        plt.pcolormesh(u[:,-1].reshape((M+2,M)), cmap='jet', shading='auto')
        # plt.imshow(u[:,-1].reshape((M+2,M)), cmap = 'jet',interpolation = 'none', origin ='upper',aspect= 'auto')
        plt.colorbar(label='Temperature')
        plt.xlabel("x-axis")
        plt.ylabel("z-axis")
        plt.title("Temperature distr after 1st cooling cycle")
        plt.show()
