from utils.utils import import_PSTAR_data
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable    
from scipy import sparse,linalg,optimize
from scipy.sparse.linalg import spsolve
from utils.profiling import timer,cpuprf

### Units 
MeV = 1e6
mA = 1e-3

def define_parameters (Ep = 90, 
                        material = 'Cu',
                        p_freq = 14,
                        p = 10,                  # change to 2 to see behaviour
                        t_pulse = 0.002,
                        I_beam = 65 * mA,
                        sigma_x = 2,
                        sigma_y = 2,
                        epsilon = 0.1,
                        D = 0.4,
                        L = 10,
                        T0 = 300,
                        gamma = 5.0,
                        plot_option = 1
                        ):

    """
    Basic configuration of parameters preset as default, add input to define_parameters to change setup.
    
    Parameters:
        See return values as all parameters are directly returned.

    Returns 
        ------
        Ep : float
            Proton energy
        material : string
            Target material
        p_freq : float
            Pulse frequency of proton beam
        p : int
            Number of pulses to simulate
        t_pulse : float
            Total time of 1 pulse
        I_beam : float
            Beam current in (mA)
        sigma_x : float
            Beam-size in x-direction
        sigma_y : float
            Beam-size in y-direction
        epsilon : float
            Emissivitiy of surrounding materil bounded to [0,1] 
        D : float
            Diffusitiviy constant D
        L : float
            Plate length of target (cm)
        T0 : float
            Initial Temperature (K)
        gamma : float
            Heat transfer Coefficient
        plot_option : int - [0,1]
            Option to plot, 1 to plot
    Returns
        -------
        par : tuple
            Tuple of all collected parameters specific to Beam setup and Material
    
    """
    dS_,rho_,Cp_     = import_PSTAR_data(material,Ep,0)
    ### Overwrite from report #FIXME
    ### very hardcoded
    dS_,rho_,Cp_ = 1.67,8.9,0.44
    return Ep,material,dS_,rho_,Cp_,p_freq,p,t_pulse,I_beam,sigma_x,sigma_y,epsilon,D,L,T0,gamma,plot_option

#### Material / Controlled parameters ####

Ep,material,dS_,rho_,Cp_,p_freq,p,t_pulse,I_beam,sigma_x,sigma_y,epsilon,D,L,T0,gamma,plot_option = define_parameters()
##### Physical Parameters ##### 
tend = p / p_freq                   # total time (s)
rho = rho_                          # Blades-Nickel (g/cm^3)
C_p = Cp_                           # Blades-Nickel (J/g/K)
sigma = 5.67037442e-4               # 1e8 * Stefan-Boltzmann constant (W/((cm)^2 K^4)
#epsilon = 0.1                      # Blades-Nickel (nounit)
A = L**2                            # Area of impact (cm^2)
t = 0.1                             # thickness of the material (cm) "Wild Guess" #FIXME
V = A * t                           # Volume (cm^3) 
dEdx = dS_ * MeV                    # Blades-Nickel (eV cm^2/g)
Ib = I_beam                         # Beam peak current
t_pulse = t_pulse                   # Pulse duration (s)
t_pause = (tend - p*t_pulse) / p     # pulse "pause" duration


def set_nums(M = 50, N = 50, maxit = 100, tol = 1e-13):
    """
    
    Returns : 
        h (float) : Spatial time step
        k_pulse : Temporal time step (for fined pulse time)
        k_pause : Temporal time step for coarse pause time
        maxit(int) : Maximum iterations for Newton-Raphson algorithm (Added to have a certain stopping criterion)
        tol(float) : Tolerance set for iterative stopping criteria, mainly for Newton-Raphson iterations
        M (int) : Number of spatial grid points for spatial discretization
        N (int) : Number of temporal grid points for time discretization
    """
    
    h = L /(M+1)
    kpulse = t_pulse/N
    kpause = t_pause/N
    return h,kpulse,kpause,maxit,tol,M,N
### Get Numerical Parameters
h,k_pulse,k_pause,maxit,tol,M,N = set_nums()

par=[M,h,N,0,maxit,tol,T0,L,tend,rho,C_p,D,sigma,epsilon,A,V,dEdx,Ib,sigma_x,sigma_y,gamma];

def use_parameters(*indices):
    """
    Decorator function that selects a certain subset of parameters and uses them in decorated function

    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            ### check if a parameteter index given does not exist in the parameter vector
            par = None
            data_vars = args
            if "par" in kwargs:
                par = kwargs.pop("par")
            elif len(args) > 0 and hasattr(args[-1], "__len__"):
                if len(args[-1]) > max(indices, default=0):
                    data_vars = args[:-1]
                    par = args[-1]

            if par is None:
                return func(*args, **kwargs)

            if indices and max(indices) >= len(par):
                raise IndexError(
                    f"Index {max(indices)} is out of bounds for parameter vector of length {len(p_vec)}"
                )
            sub_par = [par[i] for i in indices]
            return func(*data_vars, *sub_par, **kwargs)
        return wrapper
    return decorator

@use_parameters(0,1,9,10,11)
def DiscLaplacian(_x,M,h,rho,C_p,D) -> sparse.array[float]:
    """
    Creating the discrete Laplaceian Lh - an (M**2)x(M**2) matrix
    """
    # _x useless parameter just to match the architecture
    C = D / rho / C_p
    ve = np.ones(M)
    # I = sparse.eye_array(M,M)
    K = C/h**2 * sparse.dia_array(([ve,-2*ve,ve],[-1,0,1]),shape = (M,M))
    Lh = sparse.kronsum(K,K)    ### This calculates kron(K,I) + kron(I,K) according to documentation #CHECK
    return Lh

@use_parameters(0,1,9,10,11)
def DiscLaplacian1D(_x,M,h,rho,C_p,D) -> sparse.array[float]:
    """
    Creating discrete Laplacian Lh for 1D - a (M^1 x M^1) matrix
    """
    C = D/rho / C_p
    ve = np.ones(M)
    K = C/h**2 * sparse.dia_array(([ve,-2*ve,ve],[-1,0,1]),shape = (M,M))

    return K

@use_parameters(1,3,4,5,6,9,10,12,13,14,15,20)
def ImkF1(v,h,k,maxit,tol,T0,rho,C_p,sigma,epsilon,A,V,gamma) -> np.array:
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
    for i in range(maxit):
        # Function and derivative values
        F1u = F1.__wrapped__(u,T0,rho,C_p,sigma,epsilon,A,V,gamma)
        dF1u = dF1.__wrapped__(u,T0,rho,C_p,sigma,epsilon,A,V,gamma)
        # Step update
        du = (u-k/2*F1u.toarray()-v) / (1-k/2 * dF1u.toarray())
        u = u - du
        # Convergence check / stopping crit
        norm_du = h**(1/2)*linalg.norm(du,2)
        if norm_du < tol :
            break
    return u

@use_parameters(1,3,4,5,6,9,10,12,13,14,15,20)
def ImLhF1(v,Lh,h,k,maxit,tol,T0,rho,C_p,sigma,epsilon,A,V,gamma):
    """
    Newton-Raphson method approximation of solution to u in : 
            (I - L*k/2  k/2*F1)u = v
    This is done to approximate the inverse of nonlinear function.
    """

    if hasattr(v, "toarray"):
        u = v = v.toarray()
    else:
        u = v 

    I = sparse.eye_array(M**2,M**2)
    
    for i in range(maxit):
        F1u = F1.__wrapped__(u,T0,rho,C_p,sigma,epsilon,A,V,gamma)         
        dF1u = dF1.__wrapped__(u,T0,rho,C_p,sigma,epsilon,A,V,gamma)

        fu = (I - Lh*k/2)@u -k/2*F1u - v
        

        D_array = sparse.diags_array(dF1u.toarray(), offsets=0, format='csr')
        jacobian = (I - Lh*k/2 - k/2 * D_array)

        du = spsolve(jacobian,fu)
        u = u - du
        norm_du = h**(1/2)*linalg.norm(du,2)                    # Factor of h**1/2 wrong ? TODO
        if norm_du < tol:
            break
    
    return u
        

@use_parameters(6,9,10,12,13,14,15,20)
def F1(v,T0,rho,C_p,sigma,epsilon,A,V,gamma) -> sparse.array:
    """
    Evaluates F1 -radiation term given by Stefan-Boltzmann Law

    Params: 
    
    Returns:

    """
    if hasattr(v, "toarray"):
        v = v.toarray()
    C = sigma*epsilon*A/V/rho/C_p 
    c_newton = gamma * A /rho /C_p/V
    T0 = float(T0)
    u = -(C*(((v+T0)/100.0)**4-(T0/100.0)**4)) - c_newton * v
    return sparse.csr_array(u)

@use_parameters(6,9,10,12,13,14,15)
def dF1(v,T0,rho,C_p,sigma,epsilon,A,V,gamma) -> sparse.array:
    """
    Evaluates the derivate wrt to v of F1(v)-radiation term

    """
    if hasattr(v, "toarray"):
        v = v.toarray()
    C = sigma*epsilon*A/V/rho/C_p
    c_newton = gamma*A/rho/C_p/V
    u = -4*C*((v + T0)/100)**3 * 1/100   ### Element wise power ### extra factor 100 here
    # print(type(u), "Tihs is the type df1")
    u = u - c_newton
    return sparse.csr_array(u)

@use_parameters(0,1,7,9,10,16,17,18,19)
def F2(s,M,h,L,rho,C_p,dEdx,Ib,sigma_x,sigma_y):
    """
    Evaluates the F2 - Pulse Reaction term given
    F2(t,x,y) = dP(t,x,y)/rho/C_p as stated in [CT, (6.3)]
    s is the "local" time in the pulse, i.e. s \belongsto [0,tpulse] 
    """
    ### TODO: s parameter is not used ? 
    ### SO can F2 just be a big fat constant ?

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
    rx = 0
    ry = 0
    cx = L/2 + rx
    cy = L/2 + ry

    dP = rho * dEdx * Ib / sigma_x / sigma_y/2/np.pi * np.exp(-((x-cx)**2/2/sigma_x**2)-((y-cy)**2/2/sigma_y**2))
    u = np.reshape(dP,(M**2,)) / rho / C_p
    return sparse.csr_array(u)


"""
This function is misplace right now
"""
@use_parameters(6,7,9,10,11,12,13,14,15,16,17,18,19,20)
def compute_coeffs(n_dU,T_end,T0,L,rho,C_p,D,sigma,epsilon,A,V,dEdx,Ib,sigma_x,sigma_y,gamma):

    # Testing for various \Delta U Values

    dU = np.linspace(0.0,float(T_end),n_dU)
    dU[0] += 1
    # print(dU)
    t = A / V

    ### Could write these c_bolt and c_newton depending on \epsilon and \gamma ? 
    ### 
    coeffs = {"dU" : dU.tolist(),
              "c_pulse" : (L**4 * dEdx * Ib / ( D**2*dU*C_p*2*np.pi*sigma_x*sigma_y)).tolist(),
              "c_Bolt" : (sigma * dU**3 * L ** 2 / (rho * C_p * t / D)).tolist(),
              "c_newton" :[ L**2 / (rho*C_p*t*D)] * len(dU),             # Does not depened on dU explicitly
              "Theta0" : (T0 / dU).tolist()
              }
    coeffs["c_laplace"] = [1]*len(dU) 
    return coeffs

@cpuprf
@timer(metric="monotonic")
def modPRSolver(u0 : np.array,Laplaceh : sparse.dia_array,F1 : Callable[...,sparse.csr_array],F2 : Callable[...,sparse.csr_array],par):
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
    u = np.zeros((M**2,p*2*N+1), dtype=np.float32)
    u[:,0] = u0
    Lh = Laplaceh
    I = sparse.eye_array(M**2,M**2)
    print("----- Modified Peaceman Rachford solver -----")
    print("--- Progress counter  ---")

    for i in range(p):
        print(f"Pulse iteration #{i+1}")

        w = np.zeros((M**2,2*N+1))
        w[:,0] = u[:,(i)*2*N]
        

        # Pulse step #i
        par[3] = k = k_pulse

        for n in range(N):
            wn = w[:,n]
            rhs = (I + k/2*Lh)@wn+k/2*(F2((n-1)*k,par)+F2(n*k,par))
            w[:,n+1] = spsolve(I-k/2*Lh,rhs)

            # wn = w[:,n]
            # rhs = (I + k/2*Lh)@wn+k/2*(F2((n-1)*k,par)+F2(n*k,par))
            # v = spsolve(I-k/2*Lh,rhs)
            # # res = ImkF1(v,par)
            # w[:,n+1] = res.reshape(M**2,)
                        

        # Pause step #i
        par[3] = k = k_pause
        for n in range(N,2*N):
            ### Convert to csr to improve flops/sec
            wn = w[:,n]
            v = (I+k/2*Lh)@(spsolve((I-k/2 * Lh),wn +k/2*F1(wn,par)))
            w[:,n+1] = ImkF1(v,par).reshape(M**2,)

        print(f"Max Temperature is under pulse is : {np.max(w)}")
        u[:,((i)*2*N+1):((i+1)*2*N+1)] = w[:,1:2*N+1]

    return u
@cpuprf
@timer(metric="monotonic")
def TrapBasic(u0,Laplaceh,F1,F2,par):
    """
    Solves the system without splitting, using a simple Trapezoidal time stepping scheme

    Params:
        u_0 (np.array) (M^2,) : Initial temperature array to initizialize the solves 
        Laplaceh (sparse.dia_array) : Discrete Laplacian of the system - a (M**2)x(M**2) matrix.
        F1 (function) : Non-linear reaction function
        F2 (function) : Time-dependant reaction function (Beam term)
        par (list) : Vector of all parameters.      
    Returns:

    """
    u = np.zeros((M**2,p*2*N+1), dtype=np.float32)
    u[:,0] = u0
    Lh = Laplaceh
    I = sparse.eye_array(M**2,M**2)

    print("----- Trapezoidal Basic Solver -----")
    print("--- Progress counter  ---")
    for i in range(p):
        print(f"Pulse iteration #{i+1}")

        ### TODO : Still unsure if w and u could be sparse / not
        w = np.zeros((M**2,2*N+1))
        w[:,0] = u[:,(i)*2*N]

        # Pulse step #i
        par[3] = k = k_pulse

        for n in range(N):
            wn = w[:,n]
            rhs = (I + k/2*Lh)@wn + k*F2(n*k,par)
            w[:,n+1] = spsolve(I - Lh*k/2,rhs)
           

        # Pause step #i
        par[3] = k = k_pause
        for n in range(N,2*N):
            wn = w[:,n]
            rhs = (I + k/2*Lh)@wn + k/2 *F1(wn,par)
            w[:,n+1] = ImLhF1(rhs,Lh,par)

        
        print(f"Max Temperature is under pulse is : {np.max(w)}")
        u[:,((i)*2*N+1):((i+1)*2*N+1)] = w[:,1:2*N+1]

    return u

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
        f"Stopping power dEdx: {dEdx:.2E}\n",
        f"Peak current Ib : {Ib}\n",
        f"Sigmax : {sigma_x} \n",
        f"Sigmay : {sigma_y} \n",
        f"T0 : {T0}\n",)


    
    # Creating Linear operators I and discrete Laplaceian Lh
    Lh = DiscLaplacian(1,par)
    
    # Initial value u_0
    u0 = np.zeros(M**2)
    # Running modified PR and IE solver to compare
    print(f"CFL Number for simulation: CFL = {k_pause/(h**2)}")
    u = TrapBasic(u0,Lh,F1,F2,par,cpuprf = True)    
    u2 = modPRSolver(u0,Lh,F1,F2,par,cpuprf = True)
    ####   
    ##### Temporary plot to see results ####

    # Arbitrarily chosen temperature vector  
    uN = u[:,N+1]
    u2N = u2[:,N+1]

    # # Absoltute and rel error
    err2Norm = linalg.norm(uN - u2N, 2) * h 
    print(f"Timestep are kpulse:{k_pulse} and kpause = {k_pause}")
    print(f"Error at Time step N: {err2Norm}")
    relerr = err2Norm / np.linalg.norm(uN)
    print(f"Relative L2 Error: {relerr:.4e}")

    xs = np.linspace(0,L,M+2)
    x,y = np.meshgrid(xs,xs)

    mat = np.zeros((M+2,M+2), dtype = np.float32)
    mat[1:M+1,1:M+1] = u[:,-1].reshape(M,M)
    
    if plot_option:
        u_yt = u.reshape((M,M,-1)) [ M//2 ,: , :] # will have shape [M,N]
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
   