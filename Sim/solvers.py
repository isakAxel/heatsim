import numpy as np
import jax 
from numba import njit, prange
import scipy.sparse as sparse
import jax.numpy as jnp
from jax import lax

"""
    Module that involves "own-made" solvers, started to compare specific algorithms for the laplacian structure

"""

### Algo 4.7 for Toeplitz ? 

jax.config.update("jax_enable_x64", True)

@njit(parallel=True,cache = True)
def SymmSolver(s,diag,rhs,Mx,Mz):
    """
    Tridiagonal solver that uses a symmetric matrix, hence only 1 superdiagonal is stored and used.

    Params:
        s(Mz,1)
        diag(Mz*Mx,1)
        upper(Mx,1)
    """
    x = np.empty(Mx*Mz)
    y = np.empty(Mx*Mz)
    rinv = np.empty(Mx*Mz)
    
    for j in prange(Mx):
        off = Mz*j
        rinv[off] = 1.0 / diag[off]
        y[off] = rhs[off]
        for i in range(1,Mz):
            t = s[i-1] * rinv[off + i - 1]
            rinv[off+i] = 1.0 / (diag[off + i] - t * s[i - 1])
    
            y[i + off] = rhs[i + off] - t * y[off + i-1]

        x[off+Mz-1] = y[off +Mz -1] * rinv[off + Mz - 1]
        for i in range(Mz - 2, -1, -1):
            x[i + off] = (y[off + i] - s[i] * x[off + i + 1]) * rinv[off + i]
    return x

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
        ## TODO if change setting the updated diag value will be a fixed index, only changing factor of D
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
def thomasMethodJAX(lower, diag, upper, rhs, Mx,My,Mz):
    """
        Thomas method that solves for rhs

        index = (ix*My + iy)*(Mz+2) + iz
    Params:
        lower (Mz,1)
        diag (Mz*Mx,1)
        upper (Mx,1)
    """
    x = np.empty(Mx*My*Mz)

    for j in prange(Mx*My):
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

@jax.jit(static_argnames=('Mx','My','Mz'))
def thomasMethodJ2(lower, diag, upper, rhs,Mx,My,Mz):
    """
        Thomas method that solves for rhs with Jax

        index = (ix*My + iy)*(Mz+2) + iz
    Params:
        lower (Mz,1)
        diag (Mz*Mx,1)
        upper (Mx,1)
    """
    x = jnp.empty(Mx*My*Mz)

    for j in range(Mx * My):
        base = j * Mz
        cp = jnp.zeros(Mz - 1, dtype=rhs.dtype)
        dp = jnp.zeros(Mz, dtype=rhs.dtype)

        cp = cp.at[0].set(upper[0] / diag[base])
        dp = dp.at[0].set(rhs[base] / diag[base])

        for i in range(1, Mz):
            denom = diag[base + i] - lower[i - 1] * cp[i - 1]
            if i < Mz - 1:
                cp = cp.at[i].set(upper[i] / denom)
            dp = dp.at[i].set((rhs[base + i] - lower[i - 1] * dp[i - 1]) / denom)

        x = x.at[base + Mz - 1].set(dp[Mz - 1])
        for i in range(Mz - 2, -1, -1):
            x = x.at[base + i].set(dp[i] - cp[i] * x[base + i + 1])
    return x


@jax.jit
def thomasMethodJ_zfirst(lower, diag, upper, rhs):
    """
    Thomas algorithm for N independent tridiagonal systems, done in place.
 
    diag, rhs: (Mz, N)   z is the leading axis, so row i holds entry i of all N systems
    lower, upper: (Mz-1,) A[i, i-1] and A[i, i+1], shared by all systems.  Needs Mz >= 2.
    All four arrays must have the same dtype.
 
    LU form of Thomas: diag is overwritten with the pivots, rhs with the forward-substituted
    rhs and finally with the solution. No other arrays are needed.
    """
    Mz = diag.shape[0]
 
    # forward elimination
    def forward(i, state):
        diag, rhs = state
        m = lower[i - 1] / diag[i - 1]
        diag = diag.at[i].add(-m * upper[i - 1])
        rhs = rhs.at[i].add(-m * rhs[i - 1])
        return diag, rhs
 
    diag, rhs = lax.fori_loop(1, Mz, forward, (diag, rhs))
 
    # back substitution
    rhs = rhs.at[Mz - 1].divide(diag[Mz - 1])
 
    def backward(k, rhs):
        i = Mz - 2 - k                       # i = Mz-2, ..., 0
        return rhs.at[i].set((rhs[i] - upper[i] * rhs[i + 1]) / diag[i])
 
    return lax.fori_loop(0, Mz - 1, backward, rhs)
 
 
@jax.jit(static_argnames=("Mx", "My", "Mz"))
def thomasMethodJ(lower, diag, upper, rhs, Mx, My, Mz):
    """
    Same call as before. index = (ix*My + iy)*Mz + iz, so system j = ix*My + iy owns
    diag[j*Mz:(j+1)*Mz] and rhs[j*Mz:(j+1)*Mz]. Returns x with the same layout.
    """
    N = Mx * My
    x = thomasMethodJ_zfirst(lower, diag.reshape(N, Mz).T, upper, rhs.reshape(N, Mz).T)
    return x.T.reshape(-1)


@njit(parallel=True,cache=True)
def thomasMethodJAX_test(lower, diag, upper, rhs,cp,x,Mx,Mz):
    """
        Thomas method that solves for rhs
    Params:
        lower (Mz,1)
        diag (Mz*Mx,1)
        upper (Mx,1)
        cp (Mz*Mx,1)
        x (Mz*Mx,1)

    """
    denominv = 0
    for j in prange(Mx):
        denominv = 1/ diag[0 + j*Mz]
        cp[0 + j*Mz] = upper[0] * denominv 
        x[0+j*Mz] = rhs[ 0 + j*Mz] * denominv

        for i in range(1, Mz):
            denominv  = 1 /(diag[i + j*Mz] - lower[i-1] * cp[j*Mz + i-1])
            if i < Mz - 1:
                cp[i + j*Mz] = upper[i] * denominv
            x[i + j*Mz] = (rhs[i+ j*Mz] - lower[i-1] * x[j*Mz + i-1]) * denominv
        for i in range(Mz - 2, -1, -1):
            x[i + j*Mz] = x[i + j*Mz] - cp[i + j*Mz] * x[i+1 + j*Mz]
            # TODO Move out communication could be issue
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
    