import matplotlib.pyplot as plt
import numpy as np
import os 
"""
This is a module for plotting the simulations WIP
"""

def plot_errors(res: list, methods : list, title,fname = "logerrs",ylim = None,norm = fr'$L_2$',root = f"./benchmarks/data/error_plots/NormErrs"):
    """
        Plots the errors as a function of the time step size k (k_pause).
    """
    nplots = len(res)

    fig,axes = plt.subplots(ncols = nplots, figsize = (17,8))
    
    axes = [axes] if nplots == 1 else axes
    fig.suptitle(title)
    for res,ax,met in zip(res,axes,methods):
        k = res['k_pause']

        if norm == fr'$L_2$':
            errs = res['errs']
        elif norm == fr'$L_\infty$': 
            errs = res['err_Linf']
        else:
            print("Norm is not defined/logged")
            return

        # Numpy arrays
        k = np.asarray(k)
        errs = np.asarray(errs)

        ax.loglog(k, errs, 'rx')
        ax.set_title(f"Using: {met}")

        # 2. Extract first marker as the common origin point
        x0, y0 = k[0], errs[0]
       
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

        # Change ylabel depending on chosen norm
        ylabel = rf"error {norm}-norm"
        ax.set_ylabel(ylabel)       
        ax.grid(True,which='both')
        ax.legend()

    fullpath = f"{root}/{fname}"
    fig.savefig(fullpath)
    print(f"Saved figure to: {fullpath}")


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
    print(f"Saved figure to: {fullpath}")

def plot_BeamInt3D(cfg, u_vec, title, method, mark_max=False, save_fig=False,
                   frac=0.1, step=1):
    """
    frac : only plot points hotter than frac * max(U)  (hides the cold shell)
    step : extra subsampling of the remaining points if the plot is slow
    """
    Mx, My, Mz = cfg.space.Mx, cfg.space.My, cfg.space.Mz
    hx, hy = cfg.space.hx, cfg.space.hy
    Lx, Ly, Lz = cfg.target.Lx, cfg.target.Ly, cfg.target.Lz
    N = cfg.time.N
    x = np.linspace(hx, hx * Mx, Mx)
    y = np.linspace(hy, hy * My, My)
    z = np.linspace(0, Lz, Mz + 2)

    U = u_vec.reshape((Mx, My, Mz + 2))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    Umax = U.max()

    # Keep only the hot region, then subsample if needed
    mask = U > frac * Umax
    Xs, Ys, Zs, Us = X[mask][::step], Y[mask][::step], Z[mask][::step], U[mask][::step]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Xs, Ys, Zs, c=Us, cmap="jet", s=6, alpha=0.5,
                    vmin=0, vmax=Umax, depthshade=False)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="Temperature")

    # Physical extents, with z inverted so the beam enters from the top
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(Lz, 0)

    ax.set_xlabel("x-axis [cm]")
    ax.set_ylabel("y-axis [cm]")
    ax.set_zlabel("z-axis (depth) [cm]")
    ax.set_title(title + f" Method:{method}")

    if mark_max:
        ix, iy, iz = np.unravel_index(np.argmax(U), U.shape)
        ax.scatter(x[ix], y[iy], z[iz], marker="D", s=60,
                   facecolor="white", edgecolor="black",
                   label=f"Max Temperature = {U[ix, iy, iz]:.1f}")
        ax.legend(loc="upper right")

    if save_fig:
        path = "./benchmarks/data/imgs/"
        fig.savefig(path + f"Temp3D_{method}_M{Mx}_N{N}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_Ds(x,ys, labels, title = f"Electronic stopping power"):
    fig, ax = plt.subplots()

    for y,lab in zip(ys,labels):
        ax.plot(x,y,label = lab)              # depth in µm
    ax.set_xlabel("Depth [Å]")
    ax.set_ylabel(r"$S_e$ (MeV cm$^2$/g)")
    ax.set_ylim(0,np.max(y)*1.1)
    ax.set_title(title)
    ax.legend()
    plt.show()