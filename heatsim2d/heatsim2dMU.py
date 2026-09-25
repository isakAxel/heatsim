from heatsim2d import define_parameters,set_nums,DiscLaplacian1D,use_parameters,set_xspace
from utils.profiling import timer
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse,linalg,special
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu

MeV = 1e6
# Setting params
Ep,material,dS_,rho,C_p,p_freq,p,t_pulse,Ib,sigma_x,sigma_y,epsilon,D,L,T0,gamma,plot_option = define_parameters(t_pulse = 0.0015)
k_pulse,k_pause,maxit,tol,N = set_nums(N = 200)

h,M = set_xspace(L,Mx=512)

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
            if(i>7):
                print(f"iterations i : {i}")    
            break 
    return u

@use_parameters(1,3,4,5,6,9,10,12,13,20)
def IkFqLz(v,fq,Lz,h,k,maxit,tol,T0,rho,C_p,sigma,epsilon,gamma):
    """
    newton's metohd approx of sol to u in 
        (I -k/2*Fq - k/2*Lz) u = v
    """
    if hasattr(v, "toarray"):
        u = v = v.toarray()  
    else:
        u =  v

    size = Lz.shape[0]
    I = sparse.eye_array(size,size)
    for i in range(maxit):
        # F and F_prime
        Fqu = Fq.__wrapped__(u,fq,h,T0,rho,C_p,sigma,epsilon,gamma)
        dFqu = dFq.__wrapped__(u,fq,h,T0,rho,C_p,sigma,epsilon,gamma)
        # Step update
        idx = Fqu.indices
        fdata = Fqu.data
        dFdata = dFqu.data

        Dia_Fz = sparse.coo_array((dFdata,(idx,idx)),shape=(size,size)).tocsr()
        # Calculate f(uk)
        fu = (I - k/2*Lz)@u  - v
        fu[idx] -= k/2*fdata
        # Jacobian of uk - J(uk)
        jacobian = I - k/2*Lz - k/2*Dia_Fz
        du = spsolve(jacobian,fu)
        # Stepupdate and check
        u = u - du
        norm_du = h**(1/2)*linalg.norm(du,2)
        if norm_du < tol :
            if(i>7):
                print(f"iterations i : {i}")    
            break 
        return u

# Import numerically stable exponential function from scipy.special
exp = special.expit
    
@use_parameters(0,1,7,9,10,16,17,18,19)
def F2_t(s,M,h,L,rho,C_p,dEdz,Ib,sigma_x,sigma_y):
    """
    Temporary dP(t) function
    x,z grid is hardcoded.
    uses analytical braggcurve approximation
    """

    # [h,L-h]
    xs = np.linspace(h,h*M,M)
    # [0,L]
    zs = np.linspace(0,L,M+2)#
    x,z = np.meshgrid(xs,zs)
    cx = L/2
    r0 = 3.4
    # dEdz_v = np.ones(M) * dEdz
    np.savetxt("zvec.log",z,fmt="%10.4f")
    braggc = P_bragg(z,dEdz,r0)
    plot = 0
    if plot:
        plt.plot(z,braggc,'bx')
        plt.xlabel("Beam trajectoy length")
        plt.ylabel("Relative energy deposit")
        plt.show()
    dP = Ib/C_p/sigma_x/2/np.pi/sigma_y*exp(-((x-cx)**2/2/sigma_x**2)) * braggc
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
            dEdz(np.array) :
                Stopping power profile in [MeVcm^2/g]

    """
    # Peak parameters
    sigma_z = 0.018*r0
    w = 0.15*sigma_z
    # Smoother bragg peak to capture the bragg peak
    w_physical = 0.15 * sigma_z
    w = 5*h
    w = w_physical

    peak_h = 1.0 + (peak - 1.0) * exp(-((z - r0)**2) / (2 * sigma_z**2))
    cutoff = 1.0 / (1.0 + exp((z - r0) / w))
    P_tilde = peak_h * cutoff
    # P = (D_entrance + peak) * cutoff
    return dEdz_0*P_tilde

def P_braggsmooth(z, dEdz_0, r0, peak=4.0,ramp_w= 1.0):
    
    
    sigma_z = 0.018 * r0
    
    # 1. Entrance region and rise up to the peak (z <= r0)
    entrance = 1.0 + (peak - 1.0) * exp(-((z - r0)**2) / (2 * sigma_z**2))

    # 2. Smooth post-peak ramp down to 0 over 'ramp_width' (z > r0)
    dz = (z - r0) / ramp_w
    # Quadratic ramp down: continuous at z=r0 (value=peak, derivative=0)
    ramp_down = peak * np.maximum(0.0, 1.0 - dz**2)

    # Combine pre-peak entrance and post-peak ramp
    P_tilde = np.where(z <= r0, entrance, ramp_down)

    return dEdz_0 * P_tilde

@timer()
def PRCoolingCycle(v0,Lh,fq,par):
    """
    Solves for the approximation v(N,j,i) where the point approximations are given as: v(n,j,i) approximates 
    the continous temperature difference v(t,x,y) =  u(t,x,y) - u0. v(n) is given at time t_n and 
    spatial point (x,y) = (i_hx,j_hy) by the scheme

    u_n+1 = (I-k/2F1).inv @(I+k/2Lh)@(I-k/2Lh).inv@(I+k/2F1) @ u_n
    
    The history of approximations is stored in an (Mx*Mz,N+1)-matrix, including inital value v0
    
    Parameters: 
        v0 (array (Mz*Mx,)):
            Initial value for the cooling cycle, as a flattened array
        Lh (array (MzMx**2,MzMx**2)):
            Full discretized laplacian operator
        fq (array (Mz*Mx,)):
            Boundary condition array with non-zero values for gridpoints with non-zero neumann BC
        par(list):
            Parameter vector with simulation parameters
    """
    # This splitting is probably convergent of order 2 
    M_l = Lh.shape[0]
    N = par[2]

    w = np.zeros((M_l,N+1))
    w[:,0] = v0
    I = sparse.eye_array(M_l,M_l)

    print("------ Cooling Cycle Boundary Term Split --------")
    par[3] = k = k_pause
    lu = splu((I - k/2*Lh).tocsc(), permc_spec = 'NATURAL')
    for n in range(N):
        wn = w[:,n].copy()
        ## Sparse add
        fqu = Fq(wn, fq, par)          # fqu is sparse csr vector 
        idx = fqu.indices
        wn[idx] += k/2* fqu.data
        v = (I +k/2*Lh) @ lu.solve(wn)
        w[:,n+1] = IkFq(v,fq,par)

    return w
@timer()
def PRCoolingCycle2(v0,Lx,Lz,fq,par):
    """
    
    In version v2 the operators change from (Lh / Fq) to  (Lx / (Lz + Fq_z)) respectively, so the splitting is made across the dimension.


    Solves for the approximation v(N,j,i) where the point approximations are given as: v(n,j,i) approximates 
    the continous temperature difference v(t,x,y) =  u(t,x,y) - u0. v(n) is given at time t_n and 
    spatial point (x,y) = (i_hx,j_hy) by the scheme

    u_n+1 = (I-k/2F1).inv @(I+k/2Lh)@(I-k/2Lh).inv@(I+k/2F1) @ u_n
    
    The history of approximations is stored in an (Mx*Mz,N+1)-matrix, including inital value v0
    
    Parameters: 
        v0 (array (Mz*Mx,)):
            Initial value for the cooling cycle, as a flattened array
        Lx (array (MzMx**2,MzMx**2)):
            Discretized laplacian operator for x coordinates
        Lz (array (MzMx**2,MzMx**2)):
            Discretized laplacian operator for z coordinates
        fq (array (Mz*Mx,)):
            Boundary condition array with non-zero values for gridpoints with non-zero neumann BC
        par(list):
            Parameter vector with simulation parameters
    """
    # This splitting is probably convergent of order 2 
    M_l = Lz.shape[0]
    N = par[2]

    w = np.zeros((M_l,N+1))
    w[:,0] = v0
    I = sparse.eye_array(M_l,M_l)

    print("------ Cooling Cycle Dimensional Split--------")
    par[3] = k = k_pause
    lux = splu((I - k/2*Lx).tocsc(), permc_spec = 'NATURAL')
    
    for n in range(N):
        wn = w[:,n].copy()
        ## Sparse add
        
        # Operating (I +k/2Fq(wn) + LZ wn)
        fqu = Fq(wn, fq, par)        # fqu is sparse csr vector 
        idx = fqu.indices
        wn[idx] += k/2* fqu.data
        wn = wn + k/2*Lz@wn 
        # Operating (I+k/2*Lx) @ (I - k/2 *Lx)^-1
        v = (I +k/2*Lx) @ lux.solve(wn)
        w[:,n+1] = IkFqLz(v,fq,Lz,par)

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

    Ix = sparse.eye_array(M,M)
    Iz = sparse.eye_array((M+2),(M+2))
    
    Lhx = DiscLaplacian1D(0,par)
    Lhz = DiscLaplacian1D(0,par)
    Lhz, fq = BoundaryConditionNeumann(Lhz,h,q0=1,qend=0)
    Lx = sparse.kron(Iz,Lhx, format='csr')
    Lz = sparse.kron(Lhz,Ix, format='csr')  

    # filename = "mat.log"
    # np.savetxt(filename, Lhz.toarray(), fmt="%10.4f")

    one_x = np.ones((M,1))
    one_z = np.ones((M+2,1))
    # Create t
    fq = sparse.kron(fq,one_x, format='csr')
    
    # L_2D = sparse.kron(Lhz,Ix, format='csr') + sparse.kron(Iz,Lhx, format='csr')
    L_2D = sparse.kronsum(Lhx,Lhz, format = 'csr')
    
    
    # v0 shape ((M+2)M,1)
    v0 = F2_t(0,par) * t_pulse
    plot = plot_option
    if plot:
        fig,ax = plt.subplots()

        x = np.linspace(h,L-h,M)
        z = np.linspace(0,L,M+2)

        mesh = ax.pcolormesh(x,z,v0.reshape((M+2,M)), cmap = 'jet',shading = 'auto')
        plt.colorbar(mesh,label='Temperature')
        plt.xlabel("x-axis[cm]")
        plt.ylabel("z-axis[cm]")
        tick_posx = tick_posz = np.arange(0,int(L)+1)
        ax.set_xticks(tick_posx)
        ax.set_yticks(tick_posz)


        plt.title("Heat after 1st heating cycle")
        plt.show()    


    # plt.spy(L_2D)
    # plt.show()
        
    u = PRCoolingCycle2(v0,Lx,Lz,fq,par)
    u2 = PRCoolingCycle(v0,L_2D,fq,par)

    err2Norm = linalg.norm(u2 - u, 2) * h
    print(f" error L2-norm : {err2Norm}")
    
    diff_time = t_pause  # total physical cooling time
    diff_coeff = D / (rho * C_p)
    diff_length = np.sqrt(diff_coeff * diff_time)
    print(f"Diffusion length: {diff_length:.4f} cm, grid spacing h: {h:.4f} cm")
    print(f"That's {diff_length/h:.1f} grid cells")

    k1 = max(k_pause,k_pulse)

    print(f"dt = {k1}")
    print(f"dx = {h}")
    print(f"CFL Number for simulation: CFL = {k_pause/(h**2)}")

    if plot_option:
        fig,ax = plt.subplots(figsize = (12,10))
        x = np.linspace(h,L-h,M)
        z = np.linspace(0,L,M+2)
        u_f  = u[:,-1]
        mesh = ax.pcolormesh(x,z,u_f.reshape((M+2,M)), cmap='jet', shading='auto')
        plt.colorbar(mesh ,label='Temperature')
        plt.xlabel("x-axis[cm]")
        plt.ylabel("z-axis[cm]")
        tick_posx = tick_posz = np.arange(int(L) + 1)
        ax.set_xticks(tick_posx)
        ax.set_yticks(tick_posz)

        
        # Plotting max temperature
        print(u_f.shape)

        idx = np.argmax(u_f)
        T_max = u_f[idx]

        iz_max,ix_max = np.unravel_index(idx, (M + 2, M))
        ax.plot(x[ix_max], z[iz_max], marker='D', markersize=6,
                markerfacecolor='white', markeredgecolor='black',
                label=f'Max Temperature = {T_max:.1f}')
        ax.legend(loc='upper right')

        plt.text(
            0.03, 0.95, 
            f"Time for cooling tpause = {t_pause}", 
            transform=plt.gca().transAxes,  
            fontsize=9,                   
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(
                facecolor='white',          # White background
                alpha=0.7,                  
            )
        )
        plt.title(f"Heat development after 1st cooling cycle,\n with M = {M} N = {N}")
        plt.show()
