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
                             F2_xz, plot_field,sparse,numba)
from Sim.operators import DiscLaplacian,DiscLaplacian1D
from datetime import datetime
from Sim.utils.profiling import run_timed
import Sim.visual as vis
import glob


### Artifact function
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

def CreateBaseline(v0,Lx,Lz,fq,cfg,outfile, method = TrapCoolingCycle):
    """
        Creates a baseline simulation of the given cfg object and saves it to outpath - {outfile}. 
        The default method is the trapezoidal rule.
        
    """
    
    u_baseline = method(v0,Lx,Lz,fq,cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"./benchmarks/data/{outfile}"
    res = pd.Series(u_baseline)
    ### Change directory to save
    res.to_csv(path,sep = " ", index = False, float_format= "%.17e")    
    return u_baseline

def run_convergence_sweep_MU(Ns,methodA : tuple[str,Callable], cfg: SimulationSetup,refsol = None,N_base = 0,NewBase = None):
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
    # Lx = sparse.kron(Iz, Lx, format="csr")
    # Lz = sparse.kron(Lz, Ix, format="csr")

    # Continue here
    ## 

    os.chdir(r"C:\Users\axelisaksson\Desktop\heatsim")
    
   
    v0 = F2_xz(cfg) * cfg.beam.t_pulse 
    nrun = len(Ns)
    results = {"N": np.array(Ns), "k_pause" : np.zeros(nrun), f"errs" : np.zeros(nrun),"time" : np.zeros(nrun),
               "err_Linf" : np.zeros(nrun) ,"nThreads" : np.zeros_like(Ns) }

    if refsol is None:
        print(f"Running baseline:M = {cfg.space.Mx}, N = {N_base}")
        cfg.time.N = int(N_base)
        u_baseline = CreateBaseline(v0,Lx,Lz,fq,cfg,NewBase)
    else:
        u_baseline = np.loadtxt(f"./benchmarks/data/{refsol}",skiprows = 1)
    # plot_field(cfg,u_baseline,"Temperature distribution after CC", "Trapezoidal Baseline",True,True)

    for i, N in enumerate(Ns):
        cfg.time.N = N       # mutate TemporalParams, k_pulse/k_pause follow
        print(f"N={N}  hx={cfg.space.hx}, hz={cfg.space.hz}  k_pause={cfg.time.k_pause:.4e}")
        u_A, dt, nthreads = run_timed(schemeA,v0,Lx,Lz,fq,cfg)
        
        
        results["time"][i] = dt
        results["k_pause"][i] = cfg.time.k_pause
        results["nThreads"][i] = nthreads
        if np.isinf(u_A).any() or np.isnan(u_A).any():
            results[f"errs"][i] = "INF"
        else:
            results[f"errs"][i] = linalg.norm(u_A - u_baseline, ord = 2) *cfg.space.hx**(1/2)*cfg.space.hz**(1/2)
            results[f"err_Linf"][i] = ninf =  linalg.norm(u_A - u_baseline, ord = np.inf)         ### Run with this value as well to see difference
            print("Norminf:",ninf)  
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


def PlotEfficiency(M):

    simnames = [f"2DMU_PR_DSPAR1_NsVsErrs_M{M}",f"2DMU_PR_DSPAR2_NsVsErrs_M{M}",f"2DMU_PR_DSPAR4_NsVsErrs_M{M}",f"2DMU_PR_DSPAR8_NsVsErrs_M{M}"]
    
    simnames2 = [f"2DMU_PR_BS_NsVsErrs_M{M}", f"2DMU_PR_DS_NsVsErrs_M{M}",  f"2DMU_TRAP_NsVsErrs_M{M}"]
    simnames = simnames + simnames2

    res = [read_results(sim) for sim in simnames]
    methods = ["PR-Boundary Split","PR-Dimensional Split", "Trapezoidal Rule"]
    # Para methods
    paramethods = ["1 Thread" , "2 Threads", "4 Threads", "8 Threads"]    
    methods = paramethods + methods
    vis.plot_efficiency(res,methods,"Logarithmic error over computation time for DS-Method", M,f"eff_PARA_ALLMETHODS_M{M}")

def PlotMain(M):

    simnameBS = f"2DMU_PR_BS_NsVsErrs_M{M}"
    simnameDS = f"2DMU_PR_DS_NsVsErrs_M{M}"
    simnameTrap = f"2DMU_TRAP_NsVsErrs_M{M}"

    resBS = read_results(simnameBS)
    resDS = read_results(simnameDS)
    resTrap = read_results(simnameTrap)

    ks = [ resBS["k_pause"] , resDS["k_pause"],resTrap["k_pause"]]
    errs = [resBS["errs"], resDS["errs"],resTrap["errs"]]
    # time = [resBS["time"], resDS["time"]]
    errInf = [resBS["err_Linf"],resDS["err_Linf"],resTrap["err_Linf"]]

    # plt.loglog(ks2,errs2,'rx')
    vis.plot_errors(ks,errs,["PR-Boundary Split","PR-Dimensional Split", "Trapezoidal Rule"],title = f"Logarithmic errors over varying time resolutions.",
                    fname = "PR_errs_BS_DS_Trap_M512", label = f"Spatial resolution M = {M}",ylim = [1e-7,1e-1],ylabel = f'error $L_2$-norm')
    
   
    plt.close()
    vis.plot_errors(ks,errInf,["PR-Boundary Split","PR-Dimensional Split", "Trapezoidal Rule"], title= f"Logarithmic errorrs over varying N", 
                    fname="PR_errs_BS_Linf", label = f"Spatial resolution M = {M}", ylim = [1e-7,1e-1],ylabel = fr'error $L_\infty$-norm')
    plt.close()
    spliterrDS = resTrap['errs'] - resDS["errs"] 

    spliterrBS = np.abs( resTrap['errs'] - resBS["errs"]) 
    ks = resBS["k_pause"]

    
    plt.loglog(ks,spliterrBS,'x',label = "splitting error BS")
    plt.loglog(ks,spliterrDS,'x',label = "splitting error DS")
    plt.legend()
    plt.grid(which= 'both')
    plt.ylabel = "Error L2norm"
    plt.xlabel = "k - step size"
    
    plt.show()

    # vis.plot_errors(ks,[spliterrBS,spliterrDS,spliterrTrap],["PR-Boundary Split","PR-Dimensional Split", "Trapezoidal Rule"],title = f"Splitting errors over time resolutions.",
                    # fname = "PR_spliterr_BS_DS_Trap_M512", label = f"Spatial resolution M = {M}" )
def CalcTime():
    simnameBS = f"2DMU_PR_BS_NsVsErrs_M{M}"
    simnameDS = f"2DMU_PR_DS_NsVsErrs_M{M}"
    simnameTrap = f"2DMU_TRAP_NsVsErrs_M{M}"

    resBS = read_results(simnameBS)
    resDS = read_results(simnameDS)
    resTrap = read_results(simnameTrap)

    sum = 0
    for res in (resBS,resDS,resTrap):
        sum += np.sum(res["time"])

    print(f"Total time for last sim was {sum/60} min")
def BenchmarkMain(nthreads = None):

    Ns = np.array([100 * i for i in range(1,16) ])
    M = 512
    cfg = build_sim(N=Ns[0], Mx = M,My= M, Mz = M, t_pulse= 1.5e-3)
    print(f"Trial for M = {M} and Ns = {Ns}")
    print("------Pause time is:", cfg.time.t_pause, "-----\n")
    
    refsol = "MU_refsol_M512_N100000"
    if nthreads: 
        print(f"Nubmer of threads: {nthreads}")
        numba.set_num_threads(nthreads)
        print("refsol= ",refsol)
            
        sim_test_name = f"2DMU_PR_DSPAR{nthreads}_NsVsErrs_M{M}"
        resultsDS = run_convergence_sweep_MU(Ns,("PR-Dimensional Split with paralellization", PRCoolingCycle2), cfg,refsol = refsol)
        resdf = save_results(sim_test_name,resultsDS)
        print(f"Saved results to {sim_test_name}\n",
            "-------- Results : ----------\n",resdf)
    
   
    
    # print("Using refsol : ", refsol)
    # sim_test_name = f"2DMU_PR_BS_NsVsErrs_M{M}"
    # resultsDS = run_convergence_sweep_MU(Ns,("PR-Boundary Split", PRCoolingCycle), cfg,refsol = refsol)
    # resdf = save_results(sim_test_name,resultsDS)
    # print(f"Saved results to {sim_test_name}\n",
    # "-------- Results : ----------\n",resdf)


    # print("refsol= ",refsol)
    # sim_test_name = f"2DMU_PR_DS_NsVsErrs_M{M}"
    # resultsDS = run_convergence_sweep_MU(Ns,("PR-Dimensional Split", PRCoolingCycle2), cfg,refsol = refsol)
    # resdf = save_results(sim_test_name,resultsDS)
    # print(f"Saved results to {sim_test_name}\n",
    #     "-------- Results : ----------\n",resdf)


    # print("refsol= ",refsol)
    # sim_test_name = f"2DMU_TRAP_NsVsErrs_M{M}"
    # results = run_convergence_sweep_MU(Ns,("Trapezoidal Rule",TrapCoolingCycle), cfg,refsol = refsol)
    # resdf = save_results(sim_test_name, results)
    # print(f"Saved results to {sim_test_name}\n",
    #     "-------- Results : ----------\n",resdf)
   

    
    # Ns = [80_000, 100_000, 120_000, 140_000, 160_000, 180_000, 200_000, 220_000]

    # sim_test_name = f"2DMU_Heun_NsVsErrs_M{M}"
    # # refsol = "MU_refsol_M512_N100000"
    # resultsDS = run_convergence_sweep_MU(Ns,("Heuns Method", HeunCoolingCyclev2), cfg,refsol = refsol)
    # resdf = save_results(sim_test_name,resultsDS)
    # print(f"Saved results to {sim_test_name}\n",
    # "-------- Results : ----------\n",resdf)

def ErrTable(M):

    methods = ["PR_DS", "PR_BS"]
    ref = read_results(f"2DMU_TRAP_NsVsErrs_M{M}").set_index("k_pause")[["errs", "err_Linf"]]

    
    pieces = {}

    for met in methods:
        simname = f"2DMU_{met}_NsVsErrs_M{M}"
        res = read_results(simname).set_index("k_pause")[["errs","err_Linf"]] 

        diff = np.abs(res - ref)
        for col in ["errs","err_Linf"]:
            pieces[(met,col)] = diff[col]

    out = pd.DataFrame(pieces)
    out.columns.names = ["method","metric"]
     
    return out


if __name__ == "__main__":   


    M = 512
    # PlotEfficiency(M)
    # PlotMain(M)
    # CalcTime()
    # nthreads = 1
    # BenchmarkMain(nthreads)
    # nthreads = 2
    # BenchmarkMain(nthreads)
    # nthreads = 4
    # BenchmarkMain(nthreads)
    # nthreads = 8
    # BenchmarkMain(nthreads)

    # simnames = [f"2DMU_PR_DSPAR1_NsVsErrs_M{M}",f"2DMU_PR_DSPAR2_NsVsErrs_M{M}",f"2DMU_PR_DSPAR4_NsVsErrs_M{M}",f"2DMU_PR_DSPAR8_NsVsErrs_M{M}"]

    PlotEfficiency(M)
    
    # err_dataFrame = ErrTable(M)
    # print(err_dataFrame)
    
    # resdf = read_results(f"2DMU_PR_BS_NsVsErrs_M{M}",f"20261008_205518", root = "./data")
    # print("new results\n",resdf)


    
