
# General Imports
import numpy as np
import tensornetwork as tn



###################################################################################################
# NUMBA CONTROL
###################################################################################################

COMPILE_WITH_NUMBA = True

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

USE_NUMBA = COMPILE_WITH_NUMBA and NUMBA_AVAILABLE
if COMPILE_WITH_NUMBA and not NUMBA_AVAILABLE:
    print("[INFO] 'numba' not found. Running without JIT. Code will run significantly slower.")
elif not COMPILE_WITH_NUMBA:
    print(
        "[INFO] JIT disabled via main.config. Running without numba. Code will run significantly"
        " slower.")



###################################################################################################
# EXACT DIAGONALIZATION
###################################################################################################

# Maximum matrix size for which to use dense diagonalization (H.toarray() + np.linalg.eigh)
_MAX_SIZE_FOR_DENSE = 3000

# Number of eigenvalues/eigenvectors to compute in sparse diagonalization (eigsh)
_NUM_EIGS_SPARSE = 150

# Energy difference threshold below which two states are considered degenerate: 
# |E1 - E2| < _E_tol_for_degeneracy
_E_TOL_FOR_DEGENERACY = 1e-10



###################################################################################################
# MONTE CALRO
###################################################################################################

# Maximum number of consecutive dead ends before raising an error
_MAX_CONSECUTIVE_DEAD_ENDS = 100

# Maximum number of consecutive stuck states before raising an error
_MAX_CONSECUTIVE_STUCKS = 100



###################################################################################################
# tensornetwork BACKEND
###################################################################################################

# Try to use JAX backend if available, otherwise fall back to numpy
try:
    import jax
    TENSORNETWORK_BACKEND = "jax"
    print("[INFO] TensorNetwork backend set to 'jax'.")
except ImportError:
    TENSORNETWORK_BACKEND = "numpy"
    print("[INFO] 'jax' not found - using 'numpy' as TensorNetwork backend instead.")