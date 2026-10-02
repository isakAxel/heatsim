from dataclasses import dataclass
from .utils import utils as ut

MeV = 1e6
mA = 1e-3
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
        dEdz = 1.67*MeV
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
    def from_target(cls,target: TargetMaterial, **kwargs):
        return cls(cx = target.Lx/2,cy = target.Ly/2, **kwargs)
    
   

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

def build_sim(
    N: int = 100,
    Ep: float = 800.0,
    material: str = 'Ni',
    # --- target ---
    Lx: float = 10.0,
    Ly: float = 10.0,
    Lz: float = 7.0,
    gamma: float = 5.0,
    # --- beam ---
    p_freq: float = 1.0,
    p: int = 1,
    t_pulse: float = 2e-3,
    Ib: float = 65e-3,
    sigmax: float = 2.0,
    sigmay: float = 2.0,
    # --- space ---
    Mx: int = 100,
    My: int = 100,
    Mz: int = 100,
) -> SimulationSetup:
    """
    Build a full SimulationSetup from individual overridable parameters.

    Any physical, beam, spatial, or temporal parameter can be overridden by
    keyword at the call site, e.g.:

        cfg = build_sim()                          # all defaults
        cfg = build_sim(N=50, Mx=50, My=50)         # finer sweep point
        cfg = build_sim(material='Cu', Ep=1000)     # different target/energy

    N controls temporal resolution (cfg.time.N) and can be swept afterward
    directly via `cfg.time.N = new_N` (k_pulse/k_pause update automatically,
    since TemporalParams is mutable and they are derived @property values).

    Mx/My/Mz control spatial resolution and are frozen on SpatialParams;
    to sweep those, build a fresh SimulationSetup (call build_sim again or
    use dataclasses.replace) rather than mutating cfg.space in place.
    """
    target = TargetMaterial.from_material(
        Ep, material, Lx=Lx, Ly=Ly, Lz=Lz, gamma=gamma,
    )

    beam = BeamParameters.from_target(
        target,
        Ep=Ep, p_freq=p_freq, p=p, t_pulse=t_pulse,
        Ib=Ib, sigmax=sigmax, sigmay=sigmay, 
    )

    space = SpatialParams(
        target.Lx, target.Ly, target.Lz,
        Mx=Mx, My=My, Mz=Mz,
    )

    time = TemporalParams(
        t_pulse=beam.t_pulse, t_pause=beam.t_pause,
        N=N
    )

    return SimulationSetup(target=target, beam=beam, space=space, time=time)


if __name__ == '__main__':

    sim = build_sim(N=100,Ep = 800,material = 'Ni')

    print(sim)
