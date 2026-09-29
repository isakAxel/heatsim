from dataclasses import dataclass
import utils.utils as ut
@dataclass(frozen = True)
class TargetMaterial:
    rho : float
    C_p : float
    Lx : float
    Ly : float
    Lz : float
    dEdz : float
    epsilon : float
    D : float
    gamma : float
    material : str = 'Al'
    
    @property
    def A(self) -> float:
        return self.Lx * self.Ly
    @property
    def V(self) -> float:
        return self.A * self.Lz
    @classmethod
    def from_material(cls, Ep, material = 'Al',Lx=10.0, Ly=10.0, Lz=5.0, gamma=5.0):
        D, rho, C_p,epsilon = ut.import_mat_data(material)
        # dEdz = ut.import_PSTAR_data(material,Ep)
        dEdz = 1.67
        return cls(rho=rho, C_p=C_p, Lx=Lx, Ly=Ly, Lz=Lz, dEdz=dEdz,
                   epsilon=epsilon, D=D, gamma=gamma, material=material)

@dataclass(frozen = True)
class BeamParameters():
    cx : float                  # [cm]
    cy : float                  # [cm]
    sigmax : float              # [cm]
    sigmay : float              # [cm]
    p_freq : float = 1          # [hz]
    p : int = 1                 # [amt]
    t_pulse : float = 2e-3      # [s]
    Ep : float = 800.0          # [MeV]
    Ib : float = 65e-3          # [A]

    Z : float = 1.0             # Using protons [1e + ]

    @property
    def tend(self):
        return self.p/self.p_freq
    @property
    def t_pause(self):
        return (self.tend - self.p*self.t_pulse) / self.p    
    @classmethod
    def from_target(cls,target: TargetMaterial, sigma = 2, **kwargs):
        return cls(cx = target.Lx/2,cy = target.Ly/2, sigmax = sigma,sigmay = sigma,**kwargs)
    
   

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
    t_pulse : float
    t_pause : float
    N : int = 100
    maxit : int = 200
    tol : float = 1e-13
   

    @property
    def k_pulse(self) -> float:
        return self.t_pulse / self.N
    @property
    def k_pause(self) -> float:
        return self.t_pause / self.N

@dataclass
class SimulationSetup:
    beam : BeamParameters
    target : TargetMaterial
    space : SpatialParams
    time : TemporalParams
    T0 : float = 300.0  # Initial temperature [K]
    sigma : float =  5.67037442e-4  # 1e8 * Stefan-Boltzmann constant [W/((cm)^2 K^4]

def build_sim(N = 100,Ep = 800.0, material ='Al', **overrides) -> SimulationSetup:
    ### Hardcoded for now
    ### Ep = 800 MeV
    target = TargetMaterial.from_material(Ep,material, Lx= 10,Ly = 10, Lz = 5, gamma = 5.0)
    beam = BeamParameters.from_target(target,Ep = Ep,**overrides)
    space = SpatialParams(target.Lx,target.Ly,target.Lz)
    time = TemporalParams(t_pulse = beam.t_pulse, t_pause = beam.t_pause,N=N)
    return SimulationSetup(target = target, beam = beam, space = space, time = time)


if __name__ == '__main__':

    sim = build_sim(N=100,Ep = 800,material = 'Ni')

    print(sim)
