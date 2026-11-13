import os
import json
from datetime import datetime
import shutil
import numpy as np
import h5py
import glob
import pickle
import yaml
# HOTFIX
_orig_load = yaml.load

def _load_with_default(stream, Loader=None):
    return _orig_load(stream, Loader=Loader or yaml.SafeLoader)
yaml.load = _load_with_default
# Fixing yaml bug

from star import ProtonSTARCalculator,ProtonMaterials
from mendeleev import element
from dotenv import load_dotenv

from srim import TRIM,Layer,Target,Ion,output
from concurrent.futures import ProcessPoolExecutor
import Sim.visual as vis
from scipy.ndimage import uniform_filter1d,gaussian_filter1d,minimum_filter1d,maximum_filter1d
"""
    Module where different types of utilities will reside, such as data-processing, fetching
    or other parts of the pipeline which is not the actual simulation
"""

# Machine-specific paths come from the .env file (see .env.example).
# load_dotenv() searches for .env starting in this file's folder and walking up;
# variables that are already set in the real environment take precedence.
load_dotenv()
 
def _env(var):
    """Value of a required setting. Looked up on use, so importing this module never fails."""
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(f"{var} is not set. Copy .env.example to .env and fill it in.")
    return value


### Material parameters are

# TODO ARTIFACT
def import_PSTAR_data(material,Ep):
    """
    Help-function to import PSTAR data 


    Params: 
        material(str):
            Material denoted by either the Element Symbol or the shortened alloy-name
        Ep(float):
            Proton-energy in MeV

    Returns:
        dS(float):
            Stopping power of target material, at interpolated values param

    """
    def get_proton_material(mat :str, protonMaterial):
        """
        Converts Element symbol to full name to ProtonMaterials enum object
        for example "Al" -> "ALUMINUM" -> ProtonMaterials.ALUMINUM
        """
        PSTARName = element(mat).name.upper()
        return getattr(protonMaterial,PSTARName)
    
    mat = get_proton_material(material,ProtonMaterials)
    calc = ProtonSTARCalculator(mat)
    dS = calc.calculate_total_stopping_powers(Ep)

    return dS



def import_thermal_data(material):


    if material in SOLIDS_TABLE:
        mat_entry = SOLIDS_TABLE[material]
        rho = mat_entry["rho"]
        c_p = mat_entry["cp"]
        k = mat_entry["k"]
        return k,rho,c_p        

    ### Mendeleev data
    el = element(material)
    k = el.thermal_conductivity  # [W/m K]
    k = k / 100          # 1m = 100cm [W/cm K]
    rho = el.density             # [g/cm**3]
    c_p = el.specific_heat_capacity # spec heat @ 25 C, 1bar [J/g/K]

    return k,rho,c_p

def import_mat_data(material):

    k,rho,c_p =import_thermal_data(material)
    eps = EMISSIVITY_TABLE[material]
    return k,rho,c_p,eps
    
def _make_layer(material, rho, Lz):
    """
    Builds the SRIM target layer.
 
    Params:
        material (str | dict):
            Either a single element symbol / name, e.g. "Al", or a full composition
            dict, e.g. SOLIDS_TABLE["SS316L"]["elements"]
        rho (float): density [g/cm^3]
        Lz (float): layer thickness [cm]
    """
    width = Lz * 1e8                                  # cm -> Å
 
    if isinstance(material, dict):
        composition = material
    else:
        composition = {
            material: {'stoich': 1.0, 'E_d': 30.0, 'lattice': 0.0, 'surface': 3.0}
        }
 
    try:
        return Layer(composition, density=rho, width=width)
    except ValueError as e:
        raise ValueError(f"Could not build SRIM layer for {material!r}: {e}") from e
 
 
def make_job_dir(i):
    """Private copy of the SRIM installation, so runs never touch the original or each other."""
    job_dir = os.path.join(_env("SIM_SRIM_RUNS_ROOT"), f"run_{i}")
    if not os.path.exists(job_dir):
        shutil.copytree(_env("SIM_SRIM_BASE"), job_dir)
    return job_dir
 
 
def calculate_stoppingpower(material, Ep, Lz, rho, ions=400):
    """
    Stopping power of protons in the target, from a single (single-threaded) SRIM run.
 
    Params:
        material (str | dict): element symbol, or composition dict (see _make_layer)
        Ep (float): proton energy [MeV]
        Lz (float): target thickness [cm]
        rho (float): density [g/cm^3]
        number_ions (int): number of protons simulated (more = less noise, but slower)
 
    Returns:
        depth (ndarray): depth grid [Å]
        Se (ndarray): stopping power at each depth [MeV cm²/g]
    """
    job_dir = make_job_dir("single")
    layer = _make_layer(material, rho, Lz)
    proton = Ion("H", energy=Ep * 1e6)                # MeV -> eV
 
    trim = TRIM(Target([layer]), proton, number_ions=ions)
    trim.run(job_dir)
 
    ioniz = output.Results(job_dir).ioniz
    depth = np.asarray(ioniz.depth)                   # [Å]
    Se = np.asarray(ioniz.ions) * 100 / rho           # eV/Å per ion -> MeV cm²/g
    return depth, Se
 
 
def run_one(args):
    """One SRIM job in its own directory (worker for calculate_stoppingpower_para)."""
    material, Ep, Lz, rho, number_ions, i = args
    job_dir = make_job_dir(i)
 
    proton = Ion("H", energy=Ep * 1e6)
    layer = _make_layer(material, rho, Lz)
 
    target = Target([layer])
    trim = TRIM(target, proton, number_ions=number_ions)
    trim.run(job_dir)
    results = output.Results(job_dir)
    return np.asarray(results.ioniz.ions)
 
def calculate_stoppingpower_para(material : dict, Ep, Lz, rho, n_jobs=8, n_workers=4,
                            ions_per_job=50):
    """
    Parallel version of calculate_stoppingpower: n_jobs independent SRIM runs of
    ions_per_job protons each, spread over n_workers processes and averaged.
    Same returns as calculate_stoppingpower.
    """
    args = [(material, Ep, Lz, rho, ions_per_job, i) for i in range(n_jobs)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        arrays = list(ex.map(run_one, args))
        avg = np.mean(arrays, axis=1)           # one max per job (works if all arrays have the same length)
        print("avg per job:", avg)    
    ions_avg = np.mean(arrays, axis=0)     # per-ion, averaged across jobs
    depth = np.asarray(output.Results(os.path.join(_env("SIM_SRIM_RUNS_ROOT"), "run_0")).ioniz.depth) 
    Se = ions_avg * 100  / rho      # eV/Å -> MeV cm²/g
    return depth, Se



"""
    Smoothing functions to get a smooth Stopping power curve
"""
def smooth_conv(y,window = 15,mode = "uniform"):
    """
        Smoothens y, with a normalized convolutional kernel (assumes uniform spacing x)
        
        Params:
            y (ndarray) : The curve y to be smoothed
            window (int) : The number of points for the kernel (forced to be odd)
    """
    if mode == "gaussian":
        return gaussian_filter1d(y,sigma = window/6, mode = "nearest")
        
    return uniform_filter1d(y,size = window, mode = "nearest")

def SoftSmooth(y, delta=2.0, window=15):
    """
    Smooth y wherever the local range (max - min) over `window` points is < delta,
    and leave it untouched elsewhere.

    delta  : same units as y (MeV cm^2/g)
    window : sweep width in points, also used as the smoothing width
    """
    y = np.asarray(y, dtype=float)

    rng = (maximum_filter1d(y, window, mode="nearest")
           - minimum_filter1d(y, window, mode="nearest"))
    flat = rng < delta                          # True where variation is "insignificant"

    base = uniform_filter1d(y, window, mode="nearest")

    # Soft 0..1 weight so there is no visible step at the boundary.
    # Eroding first keeps w > 0 only inside the flat region, so the peak never leaks in.
    w = uniform_filter1d(minimum_filter1d(flat.astype(float), 5, mode="nearest"),
                         5, mode="nearest")

    return w * base + (1 - w) * y


# -----------------
# Data handling
# -----------------

def save(cfg, v, maxT, ts, save_path=r"data/saves/", name=None):
    """
    Save the output of HeatIter to an HDF5 file.

    v    : 1D temperature rise vector, length Mx*My*(Mz+2)
    maxT : maximum temperature after each heating and cooling phase [K]
    ts   : times corresponding to the entries of maxT [s]
    """
    Mx, My, Mz2 = cfg.space.Mx, cfg.space.My, cfg.space.Mz + 2
    dT = np.asarray(v, dtype=np.float64).reshape(Mx, My, Mz2)

    os.makedirs(save_path, exist_ok=True)
    if name is None:
        name = f"heat_{datetime.now():%Y%m%d_%H%M%S}.h5"
    else:
        name = f"{name}_{datetime.now():%Y%m%d_%H%M%S}.h5"
    filepath = os.path.join(save_path, name)

    xs = np.linspace(cfg.space.hx, cfg.space.hx * Mx, Mx)
    ys = np.linspace(cfg.space.hy, cfg.space.hy * My, My)
    zs = np.linspace(0, cfg.target.Lz, Mz2)

    cfg_blob = np.frombuffer(pickle.dumps(cfg), dtype=np.uint8)

    with h5py.File(filepath, "w") as f:

        f.create_dataset("cfg_pickle", data=cfg_blob)
        d = f.create_dataset("dT", data=dT, compression="gzip", compression_opts=4)
        d.attrs["units"] = "K"
        d.attrs["axes"] = "(x, y, z)"
        d.attrs["T0"] = cfg.T0

        for key, arr in (("x", xs), ("y", ys), ("z", zs)):
            f[key] = arr
            f[key].attrs["units"] = "cm"

        f["maxT"] = np.asarray(maxT, dtype=np.float64)
        f["maxT"].attrs["units"] = "K"
        f["ts"] = np.asarray(ts, dtype=np.float64)
        f["ts"].attrs["units"] = "s"

        f.attrs["pulses_run"] = cfg.beam.p

    if filepath:
        print(f"Saved data to file {name} in folder {save_path}")

    return filepath

def readh5(fname,path = r"data/saves/"):
    
    files = glob.glob(f"{path}{fname}_*.h5")
    if not files:
        raise FileNotFoundError(f"No file matching '{fname}_*.h5' in '{path}'")
    filepath = max(files,key = os.path.getmtime)


    with h5py.File(filepath,"r") as f:

        dT = f["dT"][...]
        T0 = float(f["dT"].attrs["T0"])
        cfg = pickle.loads(f["cfg_pickle"][...].tobytes())

        data = {
            "cfg": cfg,
            "dT": dT,
            "T_final": T0 + dT,
            "T0": T0,
            "x": f["x"][...],
            "y": f["y"][...],
            "z": f["z"][...],
            "maxT": f["maxT"][...],
            "ts": f["ts"][...],
            "pulses_run": int(f.attrs["pulses_run"]),
            "filepath": filepath,
        }

    return data


EMISSIVITY_TABLE = {
    "Al" : 0.2,
    "Ni" : 0.4,
    "SS316L" : 0.23,
}

SOLIDS_TABLE = {
    # units in [g/cm3], [J/gK], [W/mK] ,[DICT w ELEMENTS]
    "SS316L" :  {
        "rho": 8, "cp": 0.5, "k": 16.3,
        "elements": {
            "Fe": {"stoich": 66.62},
            "Cr": {"stoich": 18.19},
            "Ni": {"stoich": 11.37},
            "Mo": {"stoich": 1.45},
            "Mn": {"stoich": 1.01},
            "Si": {"stoich": 0.99},
            "N":  {"stoich": 0.20},
            "C":  {"stoich": 0.09},
            "P":  {"stoich": 0.05},
            "S":  {"stoich": 0.03},
        },
    },
}

if __name__ == '__main__':
    # k,rho,cp,eps = import_mat_data("SS316L")
    # print(k,rho,cp)
    # print("epsy",eps)
    # # Ep = np.linspace(500,1000,11)
    # Ep = 800
    
    el = 'SS316L'
    material = SOLIDS_TABLE[el]["elements"]
    # el = material = "Al"
    k,rho,cp,eps = import_mat_data(el)
    
    Lz = 10
    Ep = 150
    dz,dS_LE = calculate_stoppingpower_para(material,Ep,Lz,rho,n_jobs=4)
    Ep = 800 
    dz,dS_HE = calculate_stoppingpower_para(material,Ep,Lz,rho,n_jobs=4)
    dedz = SoftSmooth(dS_HE,2,15)
    dedz2 = SoftSmooth(dS_LE,2,15)
    label = ["HE - 800 MeV", "LE - 150 MeV"]
    # label = ["gauss","uniform"]
    # vis.plot_Ds(dz,[dS_HE,dS_LE],labels,title= f"Stopping power for element {el}")
    vis.plot_Ds(dz,[dedz,dedz2],label,title= f"Stopping power for element {el}")
    
