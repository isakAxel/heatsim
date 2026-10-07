import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import linalg
from typing import Callable
from Sim.config import build_sim,SimulationSetup
import pandas as pd
from Sim.heatsim2d import TrapBasic,modPRSolver,F1,F2
from Sim.heatsim2dMU import (PRCoolingCycle, PRCoolingCycle2, TrapCoolingCycle,
                             HeunCoolingCyclev2, BoundaryConditionNeumann,
                             F2_xz, plot_field,sparse)
from Sim.operators import DiscLaplacian,DiscLaplacian1D
from datetime import datetime
from Sim.utils.profiling import run_timed
import Sim.visual as vis
import glob

def run_convergence_sweep(Ns,methodA : tuple[str,Callable], methodB : tuple[str,Callable], cfg : SimulationSetup ):
    """Run PR vs Trapezoidal for each N, return array of L2 errors."""

    nameA = methodA[0]
    nameB = methodB[0]

    schemeA = methodA[1]
    schemeB = methodB[1]

    Lh = DiscLaplacian(cfg)              # geometry/material fixed -> compute once
    u0 = np.zeros(cfg.space.Mx * cfg.space.My)
    # print(u0.shape)
    results = {"N": np.array(Ns), "k_pause" : np.zeros(len(Ns)), f"err" : np.zeros(len(Ns))}


    for i, N in enumerate(Ns):
        cfg.time.N = N                    # mutate TemporalParams, k_pulse/k_pause follow
        print(f"N={N}  h={cfg.space.hx:.4e}  k_pause={cfg.time.k_pause:.4e}")
        u_A = schemeA(u0,Lh,F1,F2,cfg)
        u_B = schemeB(u0,Lh,F1,F2,cfg)

        u_Af = u_A[:, N + 1]
        u_Bf = u_B[:, N + 1]

        results["k_pause"][i] = cfg.time.k_pause
        results[f"err"][i] = linalg.norm(u_Af - u_Bf,ord = 2) * cfg.space.hx
        save_results(f"2D_PR_TRAP_NsVsErrs_M{cfg.space.Mx}",results)
    return results

def run_convergence_sweep_MU(Ns,methodA : tuple[str,Callable], cfg: SimulationSetup,refsol = None):
    """Run PR vs Trapezoidal for each N, return array of L2 errors."""

    nameA = methodA[0]
    schemeA = methodA[1]

    Mx = cfg.space.Mx
    Mz = cfg.space.Mz
    
    Ix = sparse.eye_array(Mx, Mx)
    Iz = sparse.eye_array(Mz + 2, Mz + 2)
    
    Lx = DiscLaplacian1D(cfg,axis = "x")
    Lz = DiscLaplacian1D(cfg,axis = "z")
    Lz,fq = BoundaryConditionNeumann(Lz,q0=1,qend=0)
    
    fq = sparse.kron(fq, np.ones((Mx,1)), format = "csr")
    Lx = sparse.kron(Iz, Lx, format="csr")
    Lz = sparse.kron(Lz, Ix, format="csr")
    L_2D = (Lx + Lz).tocsr()
    ## Continue here
    ## 
    
    # TODO fix operators before running baseline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(r"C:\Users\axelisaksson\Desktop\heatsim")
    
    # u0 = np.zeros(cfg.space.Mx * cfg.space.My)
    baseline_resolution = int(1e4)
    cfg.time.N = baseline_resolution
    v0 = F2_xz(cfg) * cfg.beam.t_pulse 
    results = {"N": np.array(Ns), "k_pause" : np.zeros(len(Ns)), f"errs" : np.zeros(len(Ns)),"time" : np.zeros(len(Ns))}

    if refsol is None:
        print(f"Running baseline:M = {cfg.space.Mx}, N = {cfg.time.N}")
        u_baseline = TrapCoolingCycle(v0,Lx,Lz,fq,cfg)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"./benchmarks/data/MU_refsol_M512_N100000_2"
        res = pd.Series(u_baseline)
        ### Change directory to save
        res.to_csv(path,sep = " ", index = False, float_format= "%.4e")
    else:
        u_baseline = np.loadtxt(f"./benchmarks/data/{refsol}",skiprows = 1)
    # plot_field(cfg,u_baseline,"Temperature distribution after CC", "Trapezoidal Baseline",True,True)

    for i, N in enumerate(Ns):
        cfg.time.N = N       # mutate TemporalParams, k_pulse/k_pause follow
        print(f"N={N}  hx={cfg.space.hx}, hz={cfg.space.hz}  k_pause={cfg.time.k_pause:.4e}")
        u_A, dt = run_timed(schemeA,v0,Lx,Lz,fq,cfg)
        
        
        results["time"][i] = dt
        results["k_pause"][i] = cfg.time.k_pause
        if np.isinf(u_A).any() or np.isnan(u_A).any():
            results[f"errs"][i] = "INF"
        else:
            results[f"errs"][i] = linalg.norm(u_A - u_baseline, ord = 2) *cfg.space.hx**(1/2)*cfg.space.hz**(1/2)
        plot_field(cfg,u_A,"Tempereture distribution after CC. ",nameA,True,True)
        
    return results




def save_results(sim_test_name, results: dict, root="./benchmarks/data"):
    """
    Save sweep results (a dict of equal-length arrays, e.g. from
    run_convergence_sweep) to <root>/<sim_test_name>/err.log, and return
    them as a DataFrame.
    """
    out_dir = Path(root) / sim_test_name
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"err_{timestamp}.log"

    df = pd.DataFrame(results)
    df.to_csv(out_path, sep=" ", index=False, float_format="%.4e")

    return df

def read_results(simname, datetime = None ,root ="./benchmarks/data") -> pd.DataFrame:
    """
    Read results from table stored in root directory
    """

    if datetime:
        readstr = f"{root}/{simname}/err_{datetime}.log"
    else:
        pattern = f"{root}/{simname}/err_*.log"
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No log files found matching pattern: {pattern}")
        readstr = max(files)
    
    return pd.read_csv(readstr, sep = r"\s+", engine = "python")

# def plot_errors(ks, errs, title,**kwargs):

#     ks = np.asarray(ks)
#     errs = np.asarray(errs)
    
#     plt.loglog(ks, errs, 'rx', **kwargs)
#     plt.title(title)

#     # 2. Extract first marker as the common origin point
#     x0, y0 = ks[0], errs[0]

#     # 3. Span reference domain from first point to last point
#     x_ref = np.logspace(np.log10(x0), np.log10(ks[-1]), 100)

#     # 4. Compute constants anchored to (x0, y0)
#     c_slope1 = y0 / x0
#     c_slope2 = y0 / (x0**2)

#     # 5. Evaluate cone boundaries
#     y_slope1 = c_slope1 * x_ref
#     y_slope2 = c_slope2 * (x_ref**2)

#     # 6. Plot the fan/cone reference lines
#     plt.loglog(x_ref, y_slope1, ':', color='green', label=r"Slope 1 ($y \propto x$)")
#     plt.loglog(x_ref, y_slope2, '--', color='blue', label=r"Slope 2 ($y \propto x^2$)")
    
#     plt.xlabel("Time step size 'k_pause'")
#     plt.ylabel("error L_2 Norm")
#     plt.gca().invert_xaxis()
#     plt.grid(True,which='both')
#     if 'label' in kwargs:
#         plt.legend()    
#     plt.show()


def plot_main(M):

    simnameDS = f"2DMU_PR_DS_NsVsErrs_M{M}"
    simnameBS = f"2DMU_PR_BS_NsVsErrs_M{M}"
    simnameTrap = f"2DMU_TRAP_NsVsErrs_M{M}"

    resDS = read_results(simnameDS)
    resBS = read_results(simnameBS)
    resTrap = read_results(simnameTrap)
    ks = [ resBS["k_pause"] , resDS["k_pause"],resTrap["k_pause"]]
    errs = [resBS["errs"], resDS["errs"],resTrap["errs"]]
    # time = [resBS["time"], resDS["time"]]

    # plt.loglog(ks2,errs2,'rx')
    vis.plot_errors(ks,errs,["PR-Boundary Split","PR-Dimensional Split", "Trapezoidal Rule"],title = f"Logarithmic errors over varying time resolutions.",
                    fname = "PR_errs_BS_DS_Trap_M512", label = f"Spatial resolution M = {M}")

if __name__ == "__main__":
    

    Ns = np.array([50] + [100*i for i in range(1,11)])

    M = 512
    # plot_main(M)
    cfg = build_sim(N=Ns[0], Mx = M,My= M, Mz = M, t_pulse= 1.5e-3)
    print(f"Trial for M = {M} and Ns = {Ns}")
    print("------Pause time is:", cfg.time.t_pause, "-----\n")

    refsol = "MU_refsol_M512_N100000"
    # sim_test_name = f"2DMU_TRAP_NsVsErrs_M{M}"
    # results = run_convergence_sweep_MU(Ns,("Trapezoidal Rule",TrapCoolingCycle), cfg,refsol = refsol)
    # resdf = save_results(sim_test_name, results)

    # print(f"Saved results to {sim_test_name}\n",
    # "-------- Results : ----------\n",resdf)

    sim_test_name = f"2DMU_PR_DS_NsVsErrs_M{M}"
    # refsol = "MU_refsol_M512_N100000"
    resultsDS = run_convergence_sweep_MU(Ns,("PR-Dimensional Split", PRCoolingCycle2), cfg,refsol = refsol)
    resdf = save_results(sim_test_name,resultsDS)

    print(f"Saved results to {sim_test_name}\n",
        "-------- Results : ----------\n",resdf)

    sim_test_name = f"2DMU_PR_BS_NsVsErrs_M{M}"
    # refsol = "MU_refsol_M512_N100000"
    resultsDS = run_convergence_sweep_MU(Ns,("PR-Boundary Split", PRCoolingCycle), cfg,refsol = refsol)
    resdf = save_results(sim_test_name,resultsDS)
    print(f"Saved results to {sim_test_name}\n",
    "-------- Results : ----------\n",resdf)

    # Ns = [80_000, 100_000, 120_000, 140_000, 160_000, 180_000, 200_000, 220_000]

    # sim_test_name = f"2DMU_Heun_NsVsErrs_M{M}"
    # refsol = "MU_refsol_M512_N100000_2"
    # resultsDS = run_convergence_sweep_MU(Ns,("Heuns Method", HeunCoolingCyclev2), cfg,refsol = refsol)
    # resdf = save_results(sim_test_name,resultsDS)
    # print(f"Saved results to {sim_test_name}\n",
    # "-------- Results : ----------\n",resdf)

    
