from dataclasses import dataclass, replace
from .utils import utils as ut
@dataclass
class SimulationSetup:
    beam : BeamParameters
    material : TargetMaterial
    space : SpatialParams
    time : TemporalParams
    T0 : float = 300.0  # Initial temperature [K]
    sigma : float =  5.67037442e-4  # 1e8 * Stefan-Boltzmann constant [W/((cm)^2 K^4]

def build_config(N = 100, **overrides) -> SimulationSetup:
    ### Hardcoded for now
    ### Ep = 800 MeV
    
    material = TargetMaterial(str = 'Al', Lx= 10,Ly = 10, Lz = 5, gamma = 5.0)



@dataclass(frozen = True)
class TargetMaterial:
    material : str = 'Al'
    rho : float
    C_p : float
    Lx : float
    Ly : float
    Lz : float
    dEdz : float
    epsilon : float
    D : float
    gamma : float
    
    @property
    def A(self) -> float:
        return self.Lx * self.Ly
    @property
    def V(self) -> float:
        return self.A * self.Lz
    

@dataclass(frozen = True)
class BeamParameters():
    material : str = 'Al'
    p_freq : float = 1 
    p : int = 1
    Ep : float = 800.0
    Ib : float = 65e-3
    cx : float 
    cy : float
    sigmax : float
    sigmay : float
    Z : float = 1.0        # Using protons

    @property
    def tend(self):
        return self.p/self.p_freq
    @property
    def t_pause(self):
        return (self.tend - self.p*self.t_pulse) / self.p    

    def __init__(self,target : TargetMaterial):
        self.material = target.material
        self.cx = target.Lx / 2
        self.cy = target.Ly / 2

@dataclass(frozen = True)
class SpatialParams:
    Lx : float
    Ly : float
    Lz : float 
    Mx : int = 100
    My : int = 100
    Mz : int = 100

    @property
    def hx(self) -> float:
        return self.Lx /( self.Mx + 1 ) 
    @property
    def hy(self) -> float:
        return self.Ly / ( self.My + 1 )
    @property
    def hz(self) -> float:
        return self.Lz / ( self.Mz + 1 ) 
@dataclass
class TemporalParams:
    """
    Mutable as of now
    """
    N : int = 100
    maxit : int = 200
    tol : float = 1e-13
    t_pulse : float
    t_pause : float

    @property
    def k_pulse(self) -> float:
        return self.t_pulse / self.N
    @property
    def k_pause(self) -> float:
        return self.t_pause / self.N