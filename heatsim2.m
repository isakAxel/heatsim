function [T]=heatsim2(Ep,I_beam,t_pulse,pulse_frequency_,n_pulses,sigma_x,sigma_y,material,epsilon,D,L,T0,plot_option)

% Comment 240911
%
% There must be a scaling error as we basically have
%
%  du/dt = 1e-2 Laplace u + 5e-5 (u^4-2^4) + 1e5 pulse(t) 
%
% Hence, the diffusion and radiation terms are neglectable...

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Consider the heat equation 
%
%      rho C_p d/dt T(t,x,y) = D Laplace T(t,x,y) 
%                   + dP(t,x,y) - sigma epsilon A/V (T^4-T0^4)
%           
% on the time interval [0,tend] and the (spatial) square [0,L]^2. 
% The initial and surrounding temperatures are T0. 
% 
% Introducing the temperature change w=T-T0 and rewriting the equation
% gives us
%
%      d/dt w = C Laplace w + F1(w)+F2(t) in (0,tend]x(0,L)^2,
%           w = 0 on (0,tend]x{0,L}^2,  
%        w(0) = 0 on (0,L)^2,
% where 
%
%   C = D/rho C_p 
%   F1(w) = - sigma epsilon A/V/rho/C_p ((w+T0)^4-T0^4).
%   F2(t) = dP(t,x,y)/rho/C_p 
%
% We approximate the "heat change" equation by 
% Peaceman-Rachford time-stepping [OS,eq.25] and finite differences.
%
% The discretization is stated equidistand spatial grid with M interior 
% points in each spatial dimension, i.e., 
%
% the spatial grid is [h,2h,...,Mh]^2, where h=L/(M+1).
%
% As the source term F2 will be rapidly varying in time, we will shift
% between a fine temporal grid, with time step k, and a corser grid, 
% with time step K, i.e., 
%
% the temporal grid is [k,2k,...,Nk,Nk+K,...,N(k+K),...,pN(k+K)=t_end].
% Here, p denotes the number of pulses (and pauses) in the sourse term F2.
%
% The approximation u=u(n,j,i) of the temperature change w(t,x,y) at 
% time t_n and spatial point (x,y)=(ih,jh) is given by 
%
%     u(n) = (I-k/2F1)^(-1)((I+k/2Lh)(I-k/2Lh)^(-1)(I+k/2F1)u(n-1) + 
%                              (I-k/2Lh)^(-1)(k/2(F2((n-1)k)+F2(nk)))
%
% where Lh is the discrete Laplacian (matrix), n=1,...,N, and u(0)=0. 
%
% The approximation u will be organized as an (M^2)x(p(N+1))-matrix, i.e., 
% including the inital value and exluding the boundary conditions. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
MeV = 1e6 ; 
mA = 1e-3 ;


% defining the proton energy, the material and the associated stopping
% power 
if ~exist("Ep") | isempty(Ep) 
   Ep = 90; % MeV 
end

% impoer stopping power and other material properties
if ~exist("material") | isempty(material)
    material = 'Cu'  ;
end

[~,dS_, rho_, ~,Cp_] = import_PSTAR_data(material,Ep,0);

% Take Dump exemple:  
% define pulse length with max power limited to 12kW 
% 

if ~exist("pulse_frequency_") | isempty(pulse_frequency_) 
    pulse_frequency_ = 1/30;
end


if ~exist("n_pulses") | isempty(n_pulses)
    n_pulses = 10 ;
end


if ~exist("I_beam") | isempty(I_beam)
    I_beam = 65 * mA % mA 
end

if ~exist('sigma_x') |isempty(sigma_x) 
    sigma_x = 2 ;
end

if ~exist('sigma_y') |isempty(sigma_y) 
    sigma_y = 2 ;
end

if ~exist('epsilon') | isempty(epsilon) 
    epsilon = 0.1  ; 
end

if ~exist('D') | isempty(D) 
    D = 0.4 ; % thermal conductivity  
end



if ~exist('L') | isempty(L) 
    L = 10 ;
end


if ~exist('T0') | isempty(T0) 
    T0 = 300 ;
end


if ~exist('plot_option') | isempty(plot_option)
    plot_option = 0  ;
end
   

%%
% Physical parameters/constants 
% 
% Equation in W/cm^3

%
% T0=300;                  % (K)
% L=7;                     % (cm)
p=n_pulses;              % number of pulses/pauses in dP
tend=p/pulse_frequency_; % total time (s)
rho= rho_ ;              % Blades-Nickel(g/cm^3)
C_p= Cp_ ;               % Blades-Nickel(J/g/K)
% D=0.4;                   % Thermal conductivity (J/(cm s K))
sigma=5.67037442e-4;     % 1e8 times the Stefan-Boltzmann constant (W/((cm)^2 K^4))
% epsilon=0.1;             % Blades-Nickel (unitless)
A=L^2;                   % (cm^2)
thickness = 0.1 ;        % thickness of the material (cm)   
V=L^2*thickness;         % Wild guess that the blade is 0.1 cm thick... (cm^3)
dEdx=dS_ * MeV ;         % Blades-Nickel (eV cm^2/g)
Ib= I_beam;              % Beam peak current (A)      
% sigma_x=2;               % Beam size along the horizontal axis (cm)
% sigma_y=3;               % Beam size along the vertical axis (cm)     
tpuls=t_pulse ;          % Puls duration (s)

tpause=(tend-p*tpuls)/p; % Pulse "pause" duration (s)


%%
sprintf('Pulse period: %f',tend/p) 
sprintf('Pulse frequency: %f',pulse_frequency_) 
sprintf('Pulse lenght: %f',t_pulse) 
sprintf('energy: %f',Ep)  
sprintf('Time without pulse: %f',tpause)  


disp('Beam and Power')
sprintf('density rho: %f',rho)  
sprintf('specific heat Cp: %f',C_p)  
sprintf('Stopping power dEdx: %f',dEdx)  
sprintf('Peak current Ib : %f',Ib)
sprintf('Peak current Ib : %f',Ib)
sprintf('Sigmax: %f',sigma_x)
sprintf('Sigmay: %f',sigma_y)


% Numerical parameters
%%
M=50;
h=L/(M+1);
N=50;
kpulse=tpuls/N;
kpause=tpause/N;

maxit=100;
Tol=1e-13;

% All parameters/constants used in the subfunctions

par=[M,h,N,0,maxit,Tol,T0,L,tend,rho,C_p, D,sigma,epsilon,A,V,...
                dEdx,Ib,sigma_x,sigma_y];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Creating the identity matrix I and the discrete Laplaceian Lh

I=speye(M^2,M^2);
Lh=DiscLaplaceian(par);

% Computing the approximation

u=sparse(M^2,p*2*N+1);

for i=1:p

    w=sparse(M^2,2*N+1);
    w=u(:,(i-1)*2*N+1);

    % pulse #i

    par(4)=kpulse;
    
    for n=1:N 
        
            v=(I+par(4)/2*Lh)*((I-par(4)/2*Lh)\(w(:,n)+par(4)/2*F1(w(:,n),par))...
              +(I-par(4)/2*Lh)\(par(4)/2*(F2((n-1)*par(4),par)+F2(n*par(4),par))));
        
            w(:,n+1)=ImkF1(v,par);

    end

    % pause #i
    
    par(4)=kpause;

    for n=N+1:2*N 

            v=(I+par(4)/2*Lh)*((I-par(4)/2*Lh)\(w(:,n)+par(4)/2*F1(w(:,n),par)));
            w(:,n+1)=ImkF1(v,par);
    end

    u(:,((i-1)*2*N+2):(i*2*N+1))=w(:,2:2*N+1);

end

% Final output including intital value, but not the boundary conditions

T=u+T0;

% plot temperature at time tend
    xp=linspace(0,L,M+2)';
    [x,y]=meshgrid(xp,xp);
    Mat=zeros(M+2,M+2);
    Mat(2:M+1,2:M+1)=reshape(full(u(:,end)),M,M);

if plot_option==1
    figure(1)
    surf(x,y,Mat+T0)

end

%%%%%%%%%%%%%%%%%%%%%%%%
% Sub-functions
%%%%%%%%%%%%%%%%%%%%%%%%

function Lh=DiscLaplaceian(par)

% Creating the discrete Laplaceian Lh - an (M^2)x(M^2) matrix

M=par(1);
h=par(2);

C=par(12)/par(10)/par(11);

ve=ones(M,1);
I=speye(M,M);
K=C/h^2*spdiags([ve -2*ve ve],-1:1,M,M);

Lh=kron(K,I)+kron(I,K);

%%%%%%%%%%%%%%%%%%%%%%%%

function u=ImkF1(v,par)

% Approximates the solution u of the equation
%       (I-k/2F1)u = v
% by Newton's method

h=par(2);
k=par(4);
maxit=par(5);
Tol=par(6);

u=v;
norm_du=Tol;
it=0;

while it<=maxit & norm_du>=Tol 

    F1u=F1(u,par);
    Fp1u=Fp1(u,par);

    du=(u-k/2*F1u-v)./(1-k/2*Fp1u);
    u=u-du;

    norm_du=h^(1/2)*norm(du,2);
    it=it+1;

end

%%%%%%%%%%%%%%%%%%%%%%%%

function u=F1(v,par)

% Evaluates 
%       F1(v)= - sigma epsilon A/V/rho/C_p (((v+T0)/100)^4-(T0/100)^4)
%
% Note that the 1e-8 factor from epsilon is reformulated as 1/100 in the
% temperature terms in order to avoid round-off errors.
 
C=par(13)*par(14)*par(15)/par(16)/par(10)/par(11);
T0=par(7);

u=-C*(((v+T0)/100).^4-(T0/100)^4);

%%%%%%%%%%%%%%%%%%%%%%%%

function u=Fp1(v,par)

% Evaluates 
%       d/dvF1(v) = - 4 sigma epsilon A/V/rho/C_p ((v+T0)/100)^3

C=par(13)*par(14)*par(15)/par(16)/par(10)/par(11);
T0=par(7);

u=-4*C*((v+T0)/100).^3;

%%%%%%%%%%%%%%%%%%%%%%%%

function u=F2(s,par)

% Evaluates 
%       F2(t,x,y)=dP(t,x,y)/rho/C_p 
% Here, s is the "local" time in the pulse, i.e., s in [0,tpulse].

M=par(1);
h=par(2);
L=par(8);

xp=linspace(h,L-h,M)'; 
[x,y]=meshgrid(xp,xp);

rx=L/25*(2*rand-1);
ry=L/25*(2*rand-1);

% dP as stated in [CT,eq. 6.3]. 
% "Randomly" centered at (x,y)=(L/2+rx,L/2+ry)

dP=par(10)*par(17)*par(18)/par(19)/par(20)/2/pi...
        *exp(-((x-L/2+rx).^2/2/par(19)^2)-((y-L/2+ry).^2/2/par(20)^2));
        
u=reshape(dP,M.^2,1)/par(10)/par(11);

%%%%%%%%%%%%%%%%%%%%%%%%
%
% References
%
% [OS] A. Ostermann, K. Schratz, 
% ''Error analysis of splitting methods for inhomogeneous evolution equations'', 
%   Applied Numerical Mathematics 62 (2012) 1436–1446
%
% [CT]
% "Proton Beam APerture MoniTor instrument design and overview for the
%  Commissioning and Operation of ESS High Power Beam on Target"
