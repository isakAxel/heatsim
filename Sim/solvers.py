import numpy as np
from numba import njit, prange
import scipy.sparse as sparse

"""
    Module that involves "own-made" solvers, started to compare specific algorithms for the laplacian structure

"""


### Use this so can use Algo 4.3.6
def SymmLZ(Lz,Mx):
    """
        Symmetrizes Lz so faster algortihms can be used.
    """

    s = np.ones(Lz.shape[0])
    # s[0] = np.sqrt( Lz[1,0]/Lz[0,1])
    s[0] = np.sqrt(Lz[1,0] / Lz[0,1])
    s[-1] = np.sqrt(Lz[-2,-1] / Lz[-1,-2])
    B = (sparse.diags(s) @ Lz @ sparse.diags(1/s)).tocsr()
    S2D = np.repeat(s,Mx)
    
    return B, S2D


@njit(parallel=True, cache=True)
def thomasMethod(lower, diag, upper, rhs):
    # diag, rhs: (n, m) Fortran-ordered; lower, upper: (n-1,)
    n, m = diag.shape
    x = np.empty((n, m))
    for j in prange(m):
        cp = np.empty(n - 1)
        dp = np.empty(n)
        cp[0] = upper[0] / diag[0, j]
        dp[0] = rhs[0, j] / diag[0, j]
        for i in range(1, n):
            denom = diag[i, j] - lower[i-1] * cp[i-1]
            if i < n - 1:
                cp[i] = upper[i] / denom
            dp[i] = (rhs[i, j] - lower[i-1] * dp[i-1]) / denom
        x[n-1, j] = dp[n-1]
        for i in range(n - 2, -1, -1):
            x[i, j] = dp[i] - cp[i] * x[i+1, j]
    return x


# def