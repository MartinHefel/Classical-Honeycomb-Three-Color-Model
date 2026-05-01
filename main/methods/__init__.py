"""
A collections of methods:

- **hamiltonian**: Find classical (basis) states using brute force and construct the 
    Rokhsar–Kivelson Hamiltonian, which can then be diagonalized.
- **monte_carlo**: Stochastic sampling algorithms to explore degenerate states starting from an 
    initial state, including a methods to find all degenerate states from a given initial state
    by repeated sampling and duplicate filtering.
- **tensor_network**: Compute the total number of ground states by contracting a tensor network
    and estimate the number of ground states using the power method on a transfer matrix.
- **paulings_estimate**: Estimate the number of configurations using a Pauling-like approximation.
- **observables**: Compute physical observables such as correlators and the structure factor.
"""

from . import hamiltonian
from . import monte_carlo
from . import tensor_network
from . import paulings_estimate # imports from hamiltonian
from . import observables