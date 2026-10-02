import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import linalg
from typing import Callable
from Sim.config import build_sim,SimulationSetup
import pandas as pd
from Sim.heatsim2d import TrapBasic,modPRSolver,F1,F2
from Sim.heatsim2dMU import PRCoolingCycle2
from Sim.operators import DiscLaplacian,DiscLaplacian1D
from datetime import datetime
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

def run_convergence_sweep_par(Ns,methodA : tuple[str,Callable], methodB : tuple[str,Callable], par ):
    """Run PR vs Trapezoidal for each N, return array of L2 errors."""

    nameA = methodA[0]
    nameB = methodB[0]

    schemeA = methodA[1]
    schemeB = methodB[1]

    Lh = DiscLaplacian(0,par)              # geometry/material fixed -> compute once
    u0 = np.zeros(Lh.shape[0]**2)

    results = {"N": np.array(Ns), "k_pause" : np.zeros(len(Ns)), f"err_{nameA}_vs_{nameB}" : np.zeros(len(Ns))}


    for i, N in enumerate(Ns):
        par[2] = N         # mutate TemporalParams, k_pulse/k_pause follow
        print(f"N={N}  h={par[1]}  k_pause={cfg.time.k_pause:.4e}")
        u_A = schemeA(u0,Lh,F1,F2,cfg)
        u_B = schemeB(u0,Lh,F1,F2,cfg)

        u_Af = u_A[:, N + 1]
        u_Bf = u_B[:, N + 1]

        results["k_pause"] = cfg.time.k_pause
        results[f"err_{nameA}_vs_{nameB}"] = u_Af - u_Bf
        
    return results




def save_results(sim_test_name, results: dict, root="./benchmarks/results"):
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

def read_results(simname, datetime = None ,root ="./benchmarks/results") -> pd.DataFrame:
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

def plot_errors(ks, errs, title,**kwargs):

    ks = np.asarray(ks)
    errs = np.asarray(errs)
    
    plt.loglog(ks, errs, 'rx', **kwargs)
    plt.title(title)

    # 2. Extract first marker as the common origin point
    x0, y0 = ks[0], errs[0]

    # 3. Span reference domain from first point to last point
    x_ref = np.logspace(np.log10(x0), np.log10(ks[-1]), 100)

    # 4. Compute constants anchored to (x0, y0)
    c_slope1 = y0 / x0
    c_slope2 = y0 / (x0**2)

    # 5. Evaluate cone boundaries
    y_slope1 = c_slope1 * x_ref
    y_slope2 = c_slope2 * (x_ref**2)

    # 6. Plot the fan/cone reference lines
    plt.loglog(x_ref, y_slope1, ':', color='green', label=r"Slope 1 ($y \propto x$)")
    plt.loglog(x_ref, y_slope2, '--', color='blue', label=r"Slope 2 ($y \propto x^2$)")
    
    plt.xlabel("Time step size 'k_pause'")
    plt.ylabel("error L_2 Norm")
    plt.gca().invert_xaxis()
    plt.grid(True,which='both')
    if 'label' in kwargs:
        plt.legend()    
    plt.show()



def plot_main(M):
    
    sim_test_name =  f"2D_PR_TRAP_NsVsErrs_M{M}"
    results = read_results(sim_test_name)

    ks = results["k_pause"]
    errs = results["err"]
    plt.figure(figsize=(8, 6))
    # Imported data from previous simulation
    dat = np.loadtxt("./benchmarks/results/2D_PR_TRAP_NsVsErrsImport/errs.log")
    pausetime = 0.998
    errs2 = dat[:,1]
    ks2 = pausetime / dat[:,0]
    # plt.loglog(ks2,errs2,'rx')
    plot_errors(ks,errs,title = "Log Errors of Modified PR solver compared to Trapezoidal rule", 
                label = f"Spatial res: M = {M}")
    plot_errors(ks2,errs2,title = "Log Errors of Modified PR Solver compared to Trapezoidal rule", 
                label = f"Spatial res: M = {M}")
    
if __name__ == "__main__":
    
    Ns = np.array([50*i for i in range(7,16)])
    M = 512 
    # print(f"Trial for M = {M} and Ns = {Ns}")
    # sim_test_name = f"2D_PR_TRAP_NsVsErrs_M{M}"
    
    cfg = build_sim(N=Ns[0], Mx = M,My= M )
    print("Pause time is:", cfg.time.t_pause)

    # results = run_convergence_sweep(Ns,("Trap_Rule",TrapBasic),("PRMod",modPRSolver), cfg)
    # # results = read_results(sim_test_name) 
    # resdf = save_results(sim_test_name,results)
    
    # print(f"Saved results to {sim_test_name}\n",
    #       "-------- Results : ----------\n",resdf)

    plot_main(M)
