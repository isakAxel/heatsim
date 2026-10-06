import matplotlib.pyplot as plt
import numpy as np
import os 
"""
This is a module for plotting the simulations WIP
"""

def plot_errors(ks : list, errs:list,methods : list, title,fname = "logerrs",**kwargs):

    nplots = len(ks)
    fig,axes = plt.subplots(ncols = nplots, figsize = (17,8))

    for k,err,ax,met in zip(ks,errs,axes,methods):
        
        k = np.asarray(k)
        err = np.asarray(err)
    
        ax.loglog(k, err, 'rx', **kwargs)
        tit = title
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
        
        ax.set_xlabel("Time step size 'k_pause'")
        ax.set_ylabel("error L_2 Norm")
        ax.invert_xaxis()
        ax.grid(True,which='both')
        if 'label' in kwargs:
            ax.legend() 

    fig.savefig(fname)