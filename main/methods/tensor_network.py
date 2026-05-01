
# General Imports
import numpy as np
import time
import tensornetwork as tn
import itertools
import warnings

# Internal Imports
from main.config import TENSORNETWORK_BACKEND
from main.tools.printers import print_progress_to_console
from main.tools.utils import validate_type
from main.model.lattice import (
    Lattice, nested_ListWithShifts_to_nested_list, get_adjacent_loops_for_sites)

# Set tensornetwork backend
tn.set_default_backend(TENSORNETWORK_BACKEND)



###################################################################################################
# BASIC TENSORS FOR THE SITES AND PLAQUETTES
###################################################################################################

def get_delta_tensor(n_legs:int, state:bool|int=None) -> np.ndarray:
    """
    Generalized delta tensor (for sites) with ``n_legs``. Entries are 1 when all indices are equal.
    
    Parameters
    ----------
    n_legs : int
        Number of legs of the delta tensor
    state : bool or None
        If 0/1 (False/True), enforces the respective state on the site.
        If None, 0 or 1 is allowed.
    
    Returns
    -------
    tensor : np.ndarray of shape (2, 2, ..., 2) and dtype=np.int64
        Generalized delta tensor for a site. Each entry is either 0 or 1:
        - 1 if all indices along the axes are equal (and match ``state`` if specified)
        - 0 otherwise
        The number of axes corresponds to ``n_legs``, each axis has length 2 (occupancy 0 or 1).
    """
    validate_type(n_legs, int, "n_legs")
    if state is not None:
        if isinstance(state, int):
            assert state in [0,1]
        else:
            validate_type(state, bool, "state")
    shape = (2,) * n_legs
    tensor = np.zeros(shape, dtype=np.int64)
    for idx in itertools.product([0, 1], repeat=n_legs):
        if state is None:
            if all(x == idx[0] for x in idx):
                tensor[idx] = 1
        else:
            if all(x == state for x in idx):
                tensor[idx] = 1
    return tensor



def get_T_tensor(n_legs: int, m_ref: int) -> np.ndarray:
    """
    Generalized T tensor (for plaquettes) with `n_legs` legs. Entries are 1 where the sum over
    indices equals `m_ref`.

    Parameters
    ----------
    n_legs : int
        Number of legs of the T tensor
    m_ref : int
        The desired filling per plaquette
    
    Returns
    -------
    tensor : np.ndarray of shape (2, 2, ..., 2) and dtype=np.int64
        Generalized T tensor for the plaquette. Each entry is either 0 or 1:
        - 1 if the sum over the indices equals ``m_ref``
        - 0 otherwise
        The number of axes corresponds to ``n_legs``, each axis has length 2 (occupancy 0 or 1).
    """
    validate_type(n_legs, int, "n_legs")
    validate_type(m_ref, int, "m_ref")
    shape = (2,) * n_legs
    tensor = np.zeros(shape, dtype=np.int64)
    for idx in itertools.product([0, 1], repeat=n_legs):
        if sum(idx) == m_ref:
            tensor[idx] = 1
    return tensor



###################################################################################################
# NETWORK CONTRACTION | FIND ALL STATES
###################################################################################################

def get_number_of_all_states_within_manifold(
        m_ref:int,
        # explicit
        ex_pl_id_to_site_ids:list[list[int]] = None,
        # object based
        lattice:Lattice=None,
) -> tuple[int, dict]:
    r""" Find the number of states for a lattice with ``m_ref`` particles per plaquette.

    **Limitations**: This method is bounded by ``min(Nx, Ny)``, where ``Nx`` (``Ny``) is the number
    of sites in the x (y) direction. As a result, the method is limited by the shorter system
    dimension, but not directly by the total number of sites or the expected number of states,
    provided ``number_of_states`` can be stored as Python integer.

    Parameters
    ----------
    m_ref : int
        Filling per plaquette (proportional to effective chemical potential).
    
    Choose one of the following:

    **Explicit** ----------------------------------------------------------------------------------

    ex_pl_id_to_site_ids : list[list[int]]
        List of lists, where each sublist contains the site IDs of a plaquette.
        The index corresponds to the plaquette ID.
        Note that total number of sites is determined by the largest site ID in the system.

    **Object-based** ------------------------------------------------------------------------------

    lattice : Lattice
        The lattice on which the states are defined.

    Returns
    -------
    number_of_states : int
        The number of states in the manifold with ``m_ref`` particles per plaquette.
    diagnostics : dict
        A dictionary of diagnostics:
        * **walltime_build** (float): Time spent for build the tensor network (seconds).
        * **walltime_contract** (float): Time spent for contracting the tensor network (seconds).
    """

    # Parameter validation ------------------------------------------------------------------------
    validate_type(m_ref, int, "m_ref")

    # Determine input mode ------------------------------------------------------------------------
    if lattice is not None:
        validate_type(lattice, Lattice, "lattice")
        input_object_based = True
    else:
        if (ex_pl_id_to_site_ids is None):
            raise ValueError(
                "Provide either the object-based input (`lattice`) "
                "or the explicit input (`ex_pl_id_to_site_ids`). "
                "Both cannot be None.")
        validate_type(ex_pl_id_to_site_ids, [list, list, int], "ex_pl_id_to_site_ids")
        input_object_based = False

    # Prepare parameters and converting data structures -------------------------------------------
    if input_object_based:
        num_of_sites = lattice.sites.N
        num_of_plaquettes = lattice.plaquettes.N
        pl_id_to_site_ids = nested_ListWithShifts_to_nested_list(lattice.plaquettes.site_ids)
        site_id_to_adjacent_pl_ids = lattice.get_adjacent_plaquettes_for_sites()
    else:
        all_site_ids = [site_id for pl in ex_pl_id_to_site_ids for site_id in pl]
        num_of_sites = max(all_site_ids) + 1
        pl_id_to_site_ids = ex_pl_id_to_site_ids
        site_id_to_adjacent_pl_ids = get_adjacent_loops_for_sites(num_of_sites, pl_id_to_site_ids)
    
    # Build tensor network ------------------------------------------------------------------------
    t0 = time.time()
    # build the T and delta tensors for the plaquettes and sites
    tensors = []
    for site_id in range(num_of_sites):
        num_of_legs = len(site_id_to_adjacent_pl_ids[site_id])
        delta = get_delta_tensor(num_of_legs)
        tensors.append(tn.Node(delta, name=f"delta_{site_id}"))
    for pl_id in range(num_of_plaquettes):
        num_of_legs = len(pl_id_to_site_ids[pl_id])
        T = get_T_tensor(num_of_legs, m_ref)
        axis_names = [str(j) for j in pl_id_to_site_ids[pl_id]]
        tensors.append(tn.Node(T, name=f"T_{pl_id}", axis_names=axis_names))
    
    # connect T legs to delta legs
    delta_connected_legs = np.zeros((num_of_sites)) # to track the legs which are already connected
    for pl_id in range(num_of_plaquettes):
        tensor_pl_id = num_of_sites+pl_id
        for site_id in pl_id_to_site_ids[pl_id]:
            tensor_site_id = site_id
            leg_name = int(delta_connected_legs[site_id])
            # establish connection
            tensors[tensor_pl_id][str(site_id)] ^ tensors[tensor_site_id][leg_name]
            delta_connected_legs[site_id] += 1 # increase count because one is already used

    # check connectivity 
    for site_id in range(num_of_sites):
        for site_id in range(num_of_sites):
            num_of_delta_legs = len(site_id_to_adjacent_pl_ids[site_id])
            assert delta_connected_legs[site_id] == num_of_delta_legs, (
                f"All delta three delta legs of all sites mus be connected but got: "
                f"{delta_connected_legs}")
    t_build = time.time()-t0

    # Contract tensor network ---------------------------------------------------------------------
    t0 = time.time()
    final_node = tn.contractors.greedy(tensors)
    number_of_states = int(final_node.tensor)
    t_contract = time.time()-t0

    # Construct return objects --------------------------------------------------------------------
    diagnostics = {
        "walltime_build": float(t_build),
        "walltime_contract": float(t_contract)}
    return number_of_states, diagnostics



###################################################################################################
# DOMINANT EIGENVALUE OF TRANSFER MATRIX T = A*A*...A*A
###################################################################################################

def get_alpha_from_transfer_matrix(
        L:int, chi:int=30, relative_tolerance:float=1e-5, max_iterations:int=int(1e3), 
        print_progress:bool=False,
        # explicit
        ex_A_transfer:np.ndarray=None, ex_num_of_sites_per_A_transfer:int=None, 
        # object based
        lattice:Lattice=None,
        A_transfer_kwargs:dict=None,
) -> tuple[float, dict]:
    r""" 
    Find the dominant eigenvalue ``alpha`` of a ring transfer matrix:
    .. code-block:: text
        |      |         |    |           |    |
        |   -- T -- = -- A -- A -- ... -- A -- A -- 
        |      |         |    |           |    |
    
    with ``L`` transfer tensors ``A=A_transfer`` with the power method, such that the total number
    of states is given by ``omega=alpha**N_sites``.

    Parameters
    ----------
    L : int
        Length of the transfer matrix, i.e. a ring formed by ``L`` unit tensors ``A_transfer``.
    chi : int
        Bond dimension of the eigenvector.
    relative_tolerance : float
        Desired relative accuracy for the dominant eigenvalue estimate.
        Iteration stops when ``|alpha_new/alpha_old-1| < relative_tolerance``
    max_iterations : int
        Maximum number of iterations.
    print_progress : bool, default False
        Whether to print information about the progress in the console.

    Choose one of the following:

    **Explicit** ----------------------------------------------------------------------------------

    ex_A_transfer : np.ndarray
        Four leg unit tensor (labels: top, right, bottom, left) which forms the transfer matrix T.
    ex_num_of_sites_per_A_transfer : int 
        Number of sites in the unit tensor ``ex_A_transfer``.

    **Object-based** ------------------------------------------------------------------------------

    lattice : Lattice
        The lattice on which the states are defined.
    A_transfer_kwargs : dict, optional
        Keyword arguments forwarded to ``lattice.get_A_transfer(**A_transfer_kwargs)``.

    Returns
    -------
    alpha : float
        The dominant eigenvalue ``alpha``, such that the total number of states is given by 
        ``omega=alpha**N_sites``
    diagnostics : dict
        A dictionary of diagnostics:
        * **walltime_power_method** (float): Time spent for finding the eigenvalue with the power
        method (seconds).
        * **converged** (bool): Whether the power method converged before reaching the maximum 
        number of iterations.
        * **number_of_performed_iterations** (int): The number of iterations actually performed 
        by the power method.
        * **final_relative_change** (float): Relative change in the dominant eigenvalue at the 
        last iteration (useful for assessing convergence when not converged).
    """

    # Parameter validation ------------------------------------------------------------------------
    validate_type(L, int, "L")
    validate_type(chi, int, "chi")
    validate_type(relative_tolerance, float, "relative_tolerance")
    validate_type(max_iterations, int, "max_iterations")
    validate_type(print_progress, bool, "print_progress")

    # Determine input mode ------------------------------------------------------------------------
    if lattice is not None:
        validate_type(lattice, Lattice, "lattice")
        input_object_based = True
    else:
        if (ex_A_transfer is None or ex_num_of_sites_per_A_transfer is None):
            raise ValueError(
                "Provide either the object-based input (`lattice`) "
                "or the explicit input (`ex_A_transfer`, `ex_num_of_sites_per_A_transfer`). "
                "Both cannot be None.")
        validate_type(ex_A_transfer, np.ndarray, "ex_A_transfer")
        validate_type(ex_num_of_sites_per_A_transfer, int, "ex_num_of_sites_per_A_transfer")
        input_object_based = False
    
    # Prepare parameters --------------------------------------------------------------------------
    if A_transfer_kwargs is None:
        A_transfer_kwargs = {}
        
    if input_object_based:
        A_transfer, num_of_sites_per_A_transfer = lattice.get_A_transfer(**A_transfer_kwargs)
    else:
        A_transfer = ex_A_transfer
        num_of_sites_per_A_transfer = ex_num_of_sites_per_A_transfer

    # Check tensor --------------------------------------------------------------------------------
    if not (A_transfer.ndim==4):
        raise ValueError(
            f"get_alpha_from_transfer_matrix(): expected A_transfer.ndim to be 4 (leg labels top, "
            f"right, bottom, left) but got {A_transfer.ndim}")
    d_t, D_r, d_b, D_l = A_transfer.shape # legs: t, r, b, l
    if not (d_t == d_b and D_r == D_l):
        raise ValueError(
            f"get_alpha_from_transfer_matrix(): expected "
            f"d_t==d_b and D_r == D_l, where d_t, D_r, d_b, D_l = A_transfer.shape, but got"
            f"d_t, D_r, d_b, D_l = {d_t}, {D_r}, {d_b}, {D_l}")
    d, D = d_t, D_r

    # Use power method ----------------------------------------------------------------------------
    t0 = time.time()
    vec = _get_initial_vector(L, d, chi)
    i = 0
    converged = False
    while i < max_iterations:
        if print_progress:
            print_progress_to_console(i, max_iterations, num_updates=-1)
        vec_new = _apply_T_to_vector(A_transfer, vec)
        beta = _get_overlap(vec, vec_new)
        alpha_new = (beta)**(1/(num_of_sites_per_A_transfer*L))
        if i != 0:
            eps = np.abs(alpha_new/alpha-1)
            if eps < relative_tolerance:
                converged = True
                break
        alpha = alpha_new
        vec = vec_new/_get_norm(vec_new)
        i += 1

    if not converged:
        eps_rounded = np.round( eps, int( -np.log(relative_tolerance) ) )
        warnings.warn(
            f"get_alpha_from_transfer_matrix(): the maximum number of iterations ({max_iterations}"
            f") were reached with an error of {eps_rounded}.")
    
    alpha_imag = np.imag(alpha)
    alpha = float(np.real(alpha))
    imag_tolerance = 0.001*alpha
    if np.abs(alpha_imag) > imag_tolerance:
        warnings.warn(
            f"get_alpha_from_transfer_matrix(): the imaginary part of alpha is large "
            f"(|Im(alpha)| = {alpha_imag} > 0.001*alpha), returning only the real part.")

    t_power_method = time.time()-t0
    
    # Construct return objects --------------------------------------------------------------------
    diagnostics = {
        "walltime_power_method": float(t_power_method),
        "converged": converged,
        "number_of_performed_iterations": int(i),
        "final_relative_change": float(eps)}
    return alpha, diagnostics



def _get_initial_vector(L:int, d:int, chi:int) -> np.ndarray:
    M_list = []
    for i in range(L):
        M_list.append(_random_complex(d, chi)) # vL, p, vR
    state = np.tensordot(M_list[0], M_list[1], axes=[[-1],[0]]) # vL, p_0, p_1, vR
    for i in range(2, L):
        # vL, p_0, ..., p_{i-1}, vR | vL, p_i, vR -> vL, p_0, ..., p_{i-1}, p_i, vR
        state = np.tensordot(state, M_list[i], axes=[[-1],[0]])
    # vL, p_0, ..., p_{L-1}, vR
    state = np.einsum('i ... j -> ...', state)
    state /= _get_norm(state)
    return state



def _random_complex(d:int, chi:int) -> np.ndarray:
    res = 1-2*np.random.rand(chi,d,chi) + 1j*(1-2*np.random.rand(chi,d,chi))
    return res



def _apply_T_to_vector(A_transfer:np.ndarray, vector:np.ndarray) -> np.ndarray:
    assert A_transfer.ndim == 4
    d_t, D_r, d_b, D_l = A_transfer.shape # legs: t, r, b, l
    assert d_t == d_b and D_r == D_l
    d, D = d_t, D_r
    L = vector.ndim
    assert L > 1
    state_new = vector.copy()
    for i in range(L):
        if i == 0:
            # b_0, ..., b_{L-1} | t, wr, b_i, wl -> b_1, ..., b_{L-1}, wr, b_0, wl
            state_new = np.tensordot(state_new, A_transfer, axes=[[0],[0]])
            # b_1, ..., b_{L-1}, wr, b_0, wl -> b_0, ..., b_{L-1}, b_0, wl, wr
            state_new = state_new.transpose(list(range(L-1))+[-2,-1,-3])
        else:
            # b_i, ..., b_{L-1}, b_0, ..., b_{i-1}, wl, wr | t, wr, b_i, wl 
            # -> b_{i+1}, ..., b_{L-1}, b_0, ..., b_{i-1}, wl, wr, b_i
            state_new = np.tensordot(state_new, A_transfer, axes=[[0,-1],[0,3]])
            # b_{i+1}, ..., b_{L-1}, b_0, ..., b_{i-1}, wl, wr, b_i 
            # -> b_{i+1}, ..., b_{L-1}, b_0, ..., b_{i-1}, b_i, wl, wr
            state_new = state_new.transpose(list(range(L-1))+[-1,-3,-2])
    # b_0, ..., b_{L-1}, wl, wr -> b_0, ..., b_{L-1}
    state_new = np.einsum('... i j -> ...', state_new)
    return state_new



def _get_overlap(state1: np.ndarray, state2: np.ndarray) -> complex:
    assert state1.shape == state2.shape
    L = state1.ndim
    return np.tensordot(state1.conj(), state2, axes=L)



def _get_norm(state:np.ndarray) -> float:
    return np.sqrt(np.tensordot(state.conj(), state, axes=state.ndim))
