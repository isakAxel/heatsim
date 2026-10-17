import numpy as np
from numba import njit, prange
import scipy.sparse as sparse
import jax.numpy as jnp 
"""
    Module that involves "own-made" solvers, started to compare specific algorithms for the laplacian structure

"""

### Algo 4.7 for Toeplitz ? 
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

### This is quicker do to removal of a bunch of reshaping overhead,
### Try to implement JAX to get even more efficient np.array handling
@njit(parallel=True,cache=True)
def thomasMethodJAX(lower, diag, upper, rhs, Mx,Mz):
    """
        Thomas method that solves for rhs
    Params:
        lower (Mz,1)
        diag (Mz*Mx,1)
        upper (Mx,1)
    """
    x = np.empty(Mx*Mz)

    for j in prange(Mx):
        cp = np.empty(Mz - 1)
        dp = np.empty(Mz)
        cp[0] = upper[0] / diag[0 + j*Mz]
        dp[0] = rhs[ 0 + j*Mz] / diag[0 + j*Mz]
        for i in range(1, Mz):
            denom = diag[i + j*Mz] - lower[i-1] * cp[i-1]
            if i < Mz - 1:
                cp[i] = upper[i] / denom
            dp[i] = (rhs[i+ j*Mz] - lower[i-1] * dp[i-1]) / denom
        x[Mz-1 + j*Mz] = dp[Mz-1]
        for i in range(Mz - 2, -1, -1):
            x[i + j*Mz] = dp[i] - cp[i] * x[i+1 + j*Mz]
    return x


# This solver is for dense matrices need to revamp this
@njit(cache=True)
def LinSolve(L,U,rhs):
    
    n,m = L.shape
    b = rhs
    x = np.empty(n)

    for j in range(n):
        b[j] = b[j] / L[j,j]
        b[j+1:n] = b[j+1:n] - b[j] * L[j+1:n,j]

    b[n] = b[n] / L[n,n]

    for j in range(n-1,-1,2):
        b[j] = b[j]/ U[j,j]
        b[0:j] = b[0:j] - b[j] * U[0:j,j]
    b[0] = b[0] / U[0,0]

    return x
    