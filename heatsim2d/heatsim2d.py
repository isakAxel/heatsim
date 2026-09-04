from utils import import_PSTAR_data
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse,linalg,optimize
from scipy.sparse.linalg import spsolve

### Units 
MeV = 1e6
mA = 1e-3

def define_parameters (Ep = 90, 
                        material = 'Cu',
                        p_freq = 1/3,
                        p = 10,
                        t_pulse = 0.2,
                        I_beam = 65 * mA,
                        sigma_x = 2,
                        sigma_y = 2,
                        epsilon = 0.1,
                        D = 0.4,
                        L = 10,
                        T0 = 300,
                        plot_option = 1
                        ):

    """
    Basic configuration of parameters preset as default, add input to define_parameters to change setup.

    """
    dS_,rho_,Cp_ = import_PSTAR_data(material,Ep,0)
    ### Overwrite from report #FIXME
    ### very hardcoded
    dS_,rho_,Cp_ = 1.67,8.9,0.44
    return Ep,material,dS_,rho_,Cp_,p_freq,p,t_pulse,I_beam,sigma_x,sigma_y,epsilon,D,L,T0,plot_option

#### Material / Controlled parameters ####

Ep,material,dS_,rho_,Cp_,p_freq,p,t_pulse,I_beam,sigma_x,sigma_y,epsilon,D,L,T0,plot_option = define_parameters()
# eps is wrong # FIXME 
##### Physical Parameters ##### 
tend = p / p_freq                   # total time (s)
rho = rho_                          # Blades-Nickel (g/cm^3)
C_p = Cp_                           # Blades-Nickel (J/g/K)
sigma = 5.67037442e-4               # 1e8 * Stefan-Boltzmann constant (W/((cm)^2 K^4)
#eps = 0.1                          # Blades-Nickel (nounit)
A = L**2                            # Area of impact (cm^2)
t = 0.1                             # thickness of the material (cm) "Wild Guess" #FIXME
V = A * t                           # Volume (cm^3) 
dEdx = dS_ * MeV                    # Blades-Nickel (eV cm^2/g)
Ib = I_beam                         # Beam peak current
t_pulse = t_pulse                   # Pulse duration (s)
t_pause = (tend - p*t_pulse) / p     # pulse "pause" duration


def set_nums(M = 50, N = 50, maxit = 100, tol = 1e-13):
    """
    
    """
    
    h = L /(M+1)
    kpulse = t_pulse/N
    kpause = t_pause/N
    return h,kpulse,kpause,maxit,tol,M,N
### Get Numerical Parameters
h,k_pulse,k_pause,maxit,tol,M,N = set_nums()

par=[M,h,N,0,maxit,tol,T0,L,tend,rho,C_p, D,sigma,epsilon,A,V,dEdx,Ib,sigma_x,sigma_y];

def use_parameters(*indices):
    """
    Decorator function that selects a certain subset of parameters and uses them in decorated function

    """
    def decorator(func):
        @wraps(func)
        def wrapper(x,par,*args,**kwargs):
            sub_par = [par[i] for i in indices]
            return func(x,*sub_par,*args,**kwargs)
        return wrapper
    return decorator

@use_parameters(0,1,9,10,11)
def DiscLaplaceian(_x,M,h,rho,C_p,D) -> sparse.array[float]:
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

@use_parameters(1,3,4,5,6,9,10,12,13,14,15)
def ImkF1(v,h,k,maxit,tol,T0,rho,C_p,sigma,epsilon,A,V) -> np.array:
    """
    Newtons' method approximation of solution to u in:
            (I - k/2*F1)u = v
    This is done to effectively approximate the inverse, to reduce compute cost. 

    """
    # u = v

    if hasattr(v, "toarray"):
        u = v = v.toarray()  
    else:
        u =  v
        
    norm_du = tol
    ### TODO: Use scipy.optimize for this instead of implementing ones own, -> more flex
    for i in range(maxit):
        # Function and derivative values
        F1u = F1.__wrapped__(u,T0,rho,C_p,sigma,epsilon,A,V)
        dF1u = dF1.__wrapped__(u,T0,rho,C_p,sigma,epsilon,A,V)
        # Step update
        du = (u-k/2*F1u.toarray()-v) / (1-k/2 * dF1u.toarray())
        u = u - du
        # Convergence check / stopping crit
        norm_du = h**(1/2)*linalg.norm(du,2)
        if norm_du < tol :
            break
    
    return u

@use_parameters(6,9,10,12,13,14,15)
def F1(v,T0,rho,C_p,sigma,epsilon,A,V) -> sparse.array:
    """
    Evaluates F1 -radiation term given by Stefan-Boltzmann Law

    Params: 
    
    Returns:

    """
    if hasattr(v, "toarray"):
        v = v.toarray()
    C = sigma*epsilon*A/V/rho/C_p 
    T0 = float(T0)
    ### terms in f1
    # print("terms in f1")
    # print("shape, ", (((v+T0)/100.0)**4-(T0/100.0)**4).shape)
    u = (C*(((v+T0)/100.0)**4-(T0/100.0)**4))
    return sparse.csr_array(u)

@use_parameters(6,9,10,12,13,14,15)
def dF1(v,T0,rho,C_p,sigma,epsilon,A,V) -> sparse.array:
    """
    Evaluates the derivate wrt to v of F1(v)-radiation term

    """
    if hasattr(v, "toarray"):
        v = v.toarray()
    C = sigma*epsilon*A/V/rho/C_p
    u = -4*C*((v + T0)/100)**3    ### Element wise power
    # print(type(u), "Tihs is the type df1")
    
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
    rx,ry = np.random.uniform(-L/25,L/25,2)
    cx = L/2 + rx
    cy = L/2 + ry

    dP = rho * dEdx * Ib / sigma_x / sigma_y/2/np.pi * np.exp(-((x-cx)**2/2/sigma_x**2)-((y-cy)**2/2/sigma_y**2))
    u = np.reshape(dP,(M**2,)) / rho / C_p
    return sparse.csr_array(u)
    

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


    # for p in par:
    #     print(f"{p=}")

    # F2(0.01,M,h,L,rho,dEdx,Ib,sigma_x,sigma_y)
    
    # Creating Linear operators I and discrete Laplaceian Lh
    I = sparse.eye_array(M**2,M**2)
    Lh = DiscLaplaceian(1,par)

    # Computing approximation
    # Comment: using lil_array now not sure what is best
    u = sparse.lil_array((M**2,p*2*N+1), dtype=np.float32)

    # u_0 first row in u is already set since IV = [0..0]

    for i in range(p):

        print("--- Progress counter  ---")
        print(f"Pulse iteration #{i+1}")
        print(f"Max Temperature is : {np.max(u.toarray())}")

        w = sparse.lil_array((M**2,2*N+1))
        # This will need to change with 0-idx #potential BUG
        # print("w shape: ", w[:,0].toarray().shape)

        # print("u shape: ", u[:,(i-1)*2*N + 1 ].toarray().shape)
        w[:,0] = u[:,(i-1)*2*N + 1 ].toarray()

        # Pulse step #i
        k = k_pulse

        for n in range(N):
            wn = w[:,n].tocsr()
           
            rhs = wn + k/2*F1(wn,par)+k/2*(F2((n-1)*k,par)+F2(n*k,par))
            v = (I+k/2*Lh)@(spsolve(I-k/2 * Lh,rhs))
            
            res = ImkF1(v,par)
            w[:,n+1] = res.reshape(M**2,)
            

        # Pause step #i
        k = k_pause
        for n in range(N,2*N):
            ### Convert to csr to improve flops/sec
            wn = w[:,n].tocsr()
            v = (I+k/2*Lh)@(spsolve((I-k/2 * Lh),wn +k/2*F1(wn,par)))
            w[:,n+1] = ImkF1(v,par).reshape(M**2,)

       
        u[:,((i)*2*N+1):((i+1)*2*N+1)] = w[:,1:2*N+1]
        print(u[:,((i)*2*N+1):((i+1)*2*N+1)])

    ##### Temporary plot to see results ####
    
    xs = np.linspace(0,L,M+2)
    x,y = np.meshgrid(xs,xs)

    mat = np.zeros((M+2,M+2), dtype = np.float32)

    mat[1:M+1,1:M+1] = u[:,-1].toarray().reshape(M,M)

    print("types: ", type(u), u.toarray().dtype)
    plt.imshow(u.toarray(), cmap = 'hot',interpolation = 'nearest')
    plt.xlabel("Time axis")
    plt.ylabel("heat development")
    plt.show()    
    if plot_option:
        # fig,ax = plt.subplots()
        # ax.set_title(rf'Heat distribution $\delta$ T')
        plt.figure(figsize=(8,6))
        plt.imshow(mat,cmap = 'hot',interpolation= 'nearest')
        # plt.title('Heat distribution of Target material')
        plt.colorbar()
        plt.show()
   