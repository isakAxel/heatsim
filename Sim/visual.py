import matplotlib.pyplot as plt
import numpy as np
import os 
"""
This is a module for plotting the simulations WIP
"""

def plot_errors(ks : list, errs:list,methods : list, title,fname = "logerrs",ylim = None,ylabel = "error",**kwargs):

    nplots = len(ks)
    fig,axes = plt.subplots(ncols = nplots, figsize = (17,8))
    # Hotfix
    axes = [axes] if nplots == 1 else axes
    
    for k,err,ax,met in zip(ks,errs,axes,methods):
        
        k = np.asarray(k)
        err = np.asarray(err)
    
        ax.loglog(k, err, 'rx', **kwargs)

        ax.set_title(f"{title} \n Using : {met}")

        # 2. Extract first marker as the common origin point
        x0, y0 = k[0], err[0]
       
        # 3. Span reference domain from first point to last point
        x_ref = np.logspace(np.log10(x0), np.log10(k[-1]), 100)

        # 4. Compute constants anchored to (x0, y0)
        c_slope1 = y0 / x0
        c_slope2 = y0 / (x0**2)

        # 5. Evaluate cone boundaries
        y_slope1 = c_slope1 * x_ref
        y_slope2 = c_slope2 * (x_ref**2)

        # 6. Plot the fan/cone reference lines
        ax.loglog(x_ref, y_slope1, ':', color='green', label=r"Slope 1 ($y \propto x$)")
        ax.loglog(x_ref, y_slope2, '--', color='blue', label=r"Slope 2 ($y \propto x^2$)")

        if ylim:
            ax.set_ylim(ylim[0],ylim[1])
        ax.set_xlabel("Time step size 'k_pause'")

        if ylabel:
            ax.set_ylabel(ylabel)
                
        ax.grid(True,which='both')
        if 'label' in kwargs:
            ax.legend() 

    fig.savefig(fname)


def plot_efficiency(res : list, methods : list, title,M,fname = "logefficiency", root = f"./benchmarks/data/error_plots/EFFICIENCY"):
    
    """
    Plots erros over time of the algorithm from a list of results dataframes res. The produced plot is saved in the root directory.

    Parameters: 
        res(list) : 
            List of result dataframes, the dataframes contain fields [N, k_pause, errs, time].
        methods(list) : 
            List of names for methods/schemes used in the benchmark run
        title(str) :
            Title of the plot
        fname(str):
            Name of the saved image.
    """

    fig,ax = plt.subplots(figsize = (17,8))

    for r,met in zip(res,methods):
        
        time = r['time']
        errs = r['errs']

        ax.loglog(time,errs, 'x--', label = f"{met}")

        ax.set_title(f"{title} \n With spatial resolution M = {M}")

        ax.set_xlabel("Time t for simulation in seconds")
        ax.set_ylabel(rf"Error $L_2$-Norm")
        ax.grid(True, which = 'both') 
        
        ax.legend()
        
    fullpath = f"{root}/{fname}"
    fig.savefig(fullpath)