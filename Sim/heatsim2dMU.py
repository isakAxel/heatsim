import heatsim2d as hs2

MeV = 1e6
# Setting params
Ep,material,dS_,rho,C_p,p_freq,p,t_pulse,Ib,sigma_x,sigma_y,epsilon,D,L,T0,gamma,plot_option = hs2.define_parameters()
h,k_pulse,k_pause,maxit,tol,M,N = hs2.set_nums()

tend = p / p_freq
sigma  = 5.67037442e-4               # 1e8 * Stefan-Boltzmann constant (W/((cm)^2 K^4)
A = L**2
t = 0.1                             # cm
V = A * t
dEdx = dS_ * MeV                    # Blades-Nickel (eV cm^2/g)

par = [M,h,N,k_pulse,maxit,tol,T0,L,tend,rho,C_p,D,sigma,epsilon,A,V,dEdx,Ib,sigma_x,sigma_y,gamma];

print(par)