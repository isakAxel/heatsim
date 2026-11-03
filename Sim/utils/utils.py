import os
import shutil
import numpy as np

from star import ProtonSTARCalculator,ProtonMaterials
from mendeleev import element

from srim import TRIM,Layer,Target,Ion,output
from Sim.paths import srimexepath,runsrimroot,basesrim
from concurrent.futures import ProcessPoolExecutor
import Sim.visual as vis

"""
    Module where different types of utilities will reside, such as data-processing, fetching
    or other parts of the pipeline which is not the actual simulation
"""

### Material parameters are

def import_PSTAR_data(material,Ep):
    """
    Help-function to import PSTAR data 
    #TODO: #FIXME

    Params: 
        material(str):
            Material denoted by either the Element Symbol or the shortened alloy-name
        Ep(float):
            Proton-energy in MeV
        param(idk):
            idk

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
    


def make_job_dir(i):
    job_dir = os.path.join(runsrimroot, f"run_{i}")
    if not os.path.exists(job_dir):
        shutil.copytree(basesrim, job_dir)
    return job_dir

def run_one(args):
    material, Ep, Lz, rho, number_ions, i = args
    job_dir = make_job_dir(i)

    proton = Ion("H", energy=Ep * 1e6)

    if isinstance(material, dict):
        # Full composition given, e.g. SOLIDS_TABLE["SS316L"]["elements"]
        layer = Layer(material, density=rho, width=Lz * 1e8)
    else:
        # Single element or name, e.g. "Al" or "SS316L"
        try:
            layer = Layer({
                material: {'stoich': 1.0, 'E_d': 30.0, 'lattice': 0.0, 'surface': 3.0}
            }, density=rho, width=Lz * 1e8)
        except ValueError as e:
            print(f"Could not find material ({e}). Try Again")
            

    target = Target([layer])
    trim = TRIM(target, proton, number_ions=number_ions)
    trim.run(job_dir)
    results = output.Results(job_dir)
    return np.asarray(results.ioniz.ions)

def calculate_stoppingpower(material : dict, Ep, Lz, rho, n_jobs=8, n_workers=4,
                            ions_per_job=10):
    args = [(material, Ep, Lz, rho, ions_per_job, i) for i in range(n_jobs)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        arrays = list(ex.map(run_one, args))
        avg = np.mean(arrays, axis=1)           # one max per job (works if all arrays have the same length)
        print("avg per job:", avg)    
    ions_avg = np.mean(arrays, axis=0)     # per-ion, averaged across jobs
    depth = np.asarray(output.Results(os.path.join(runsrimroot, "run_0")).ioniz.depth) 
    Se = ions_avg * 100  / rho      # eV/Å -> MeV cm²/g
    return depth, Se

# TODO: Artifact
def smooth_stoppingpower(z,dS):
    """
        Takes the stopping power and the target depth and smoothens the function by using scipy.interpolate. With spline
    """
    sline = make_smoothing_spline(z,dS, lam = 100)
    return sline(z),z
# def calculate_stoppingpower(material: str,Ep,Lz,rho,plot = False, output_dir = r'/data/srim_output'):

#     proton = Ion("H",energy=Ep*1e6)

#     layerwidth = Lz # [cm]
#     layerwidthAng = layerwidth *  1e8  # [Å]
#     layer = Layer({
#         material : {
#             'stoich': 1.0,
#             'E_d': 30.0,
#             'lattice': 0.0,
#             'surface': 3.0
#         }}, density=rho, width = layerwidthAng) # [Å] -> 1e-4 [cm]
    
  
#     target = Target([layer])
#     trim = TRIM(target,proton, number_ions = 1000)
#     srim_executable_directory = srimexepath

#     trim.run(srim_executable_directory)
#     ioniz = results.ioniz

#     depth_a = ioniz.depth
#     ion_ev_per_ang = ioniz.ions
#     Se = ion_ev_per_ang * 100.0 / rho        

#     if plot:
#         vis.plot_Ds(depth_a / 1e4, Se,title= f"Stopping power for element {material}")


#     return Se
    
    
    

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
            "C":  {"stoich": 0.03},
            "Si": {"stoich": 1.0},
            "Mn": {"stoich": 2.0},
            "Pb": {"stoich": 0.045},
            "S":  {"stoich": 0.035},
            "Cr": {"stoich": 18.0},
            "Ni": {"stoich": 14.0},
            "Mo": {"stoich": 3.0},
            "N":  {"stoich": 0.1},
        },
    },
}

if __name__ == '__main__':
    # k,rho,cp,eps = import_mat_data("SS316L")
    # print(k,rho,cp)
    # print("epsy",eps)
    # # Ep = np.linspace(500,1000,11)
    # Ep = 800
    # dS = import_PSTAR_data("Al",Ep)
    # print(f"dS : {dS}")
    
    el = 'SS316L'
    material = SOLIDS_TABLE[el]["elements"]
    el = material = "Al"
    k,rho,cp,eps = import_mat_data(el)
    
    Lz = 10
    Ep = 150
    dz,dS_LE = calculate_stoppingpower(material,Ep,Lz,rho,n_jobs = 4)
    Ep = 800 
    dz,dS_HE = calculate_stoppingpower(material,Ep,Lz,rho,n_jobs = 4, )
    labels= ["HE - 800 MeV", "LE - 150 MeV"]
    vis.plot_Ds(dz,[dS_HE,dS_LE],labels,title= f"Stopping power for element {el}")

