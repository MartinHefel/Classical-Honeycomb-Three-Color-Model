
# General Imports
import numpy as np
import time
from scipy import sparse
from scipy.sparse import spmatrix
from scipy.sparse.linalg import eigsh 

# Internal Imports
from main.config import (
    USE_NUMBA, _MAX_SIZE_FOR_DENSE, _NUM_EIGS_SPARSE, 
    _E_TOL_FOR_DEGENERACY)
from main.tools.utils import validate_type, nested_list_to_padded_array
from main.tools.printers import print_progress_to_console
from main.tools.optional_njit import optional_njit
from main.model.hilbert_space import normalize_state
from main.model.lattice import Lattice, nested_ListWithShifts_to_padded_array
from main.model.states import (
    HardcoreBosonStateCollection, HardcoreBosonMBS, _get_n, _get_basis_StNr, _flip_sites_of_StNr)

# Numba types
if USE_NUMBA:
    from numba.typed import Dict
    from numba import types
else:
    Dict = None
    types = None



###################################################################################################
# ALL (CLASSICAL) BASIS STATES
###################################################################################################

def get_all_basis_states(
        m_ref:int, N_filling:int=None, V:float=1., ground_states_only:bool=False, 
        print_progress:bool=False,
        # explicit
        ex_pl_id_to_site_ids:list[list[int]] = None,
        # object based
        lattice:Lattice=None,
) -> tuple[HardcoreBosonStateCollection, dict]:
    r""" Find all basis states, including their energy, with filling ``N_filling`` on a ``lattice``
    considering the following Hamiltonian

    .. math::

        H = V \sum_{pl} [ (n_pl-m_ref)^2 - m_ref^2 ]

    **Limitations**: This method is bounded by the Hilbert space dimension ``2**num_of_sites``, 
    which determines the scaling of both CPU time and RAM usage. Consequently this approach is 
    directly limited by the system size.

    Parameters
    ----------
    m_ref : int
        Filling per plaquette (proportional to effective chemical potential).
    N_filling : int | None
        Number of particles in the system.
        If ``None``, all states are collected independent of their particle number.
        If not ``None``, only states with particle number ``N_filling`` are collected.
    V : float
        Interaction term of the Hamiltonian
    ground_states_only : bool, default False
        Whether to collect ground states only
    print_progress : bool, default False
        Whether to print information about the progress in the console.
    
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
    states : HardcoreBosonStateCollection
        The configurations on the lattice.
    diagnostics : dict
        A dictionary of diagnostics:
        * **walltime_state_collecting** (float): Time spent for collecting all states (seconds).
    """

    # Parameter validation ------------------------------------------------------------------------
    validate_type(m_ref, int, "m_ref")
    validate_type(V, float, "V")
    validate_type(ground_states_only, bool, "ground_states_only")
    validate_type(print_progress, bool, "print_progress")
    if N_filling is not None:
        validate_type(N_filling, int, "N_filling")

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
        pl_id_to_site_ids = nested_ListWithShifts_to_padded_array(
            lattice.plaquettes.site_ids, dtype=np.int64)
    else:
        all_site_ids = [site_id for pl in ex_pl_id_to_site_ids for site_id in pl]
        num_of_sites = max(all_site_ids) + 1
        pl_id_to_site_ids = nested_list_to_padded_array(ex_pl_id_to_site_ids, dtype=np.int64)

    if num_of_sites > 60:
        raise ValueError(
            f"Too many sites ({num_of_sites}) for bit representation with `dtype=np.int64`.")
    if N_filling is not None:
        if not (0 <= N_filling and N_filling < num_of_sites):
            raise ValueError(
                f"`N_filling ({N_filling})` must larger or equal 0 and smaller or equal to the "
                f"number of sites in the lattice ({num_of_sites})")
    else:
        N_filling = -1
    
    # Get all states with compiled function -------------------------------------------------------
    t0 = time.time()
    Es, StNrs = _get_all_basis_states(
        num_of_sites, pl_id_to_site_ids, m_ref, N_filling, V, ground_states_only, print_progress)
    t_state_collecting = time.time()-t0

    # Construct return objects --------------------------------------------------------------------
    states = HardcoreBosonStateCollection.from_StNrs(StNrs, num_of_sites, Es)
    diagnostics = {
        "walltime_state_collecting": float(t_state_collecting)}
    return states, diagnostics



@optional_njit()
def _get_all_basis_states(
        num_of_sites:int, pl_id_to_site_ids:np.ndarray, m_ref:int, N_filling:int,
        V:float, ground_states_only:bool, print_progress:bool
) -> tuple[np.ndarray, np.ndarray]:

    # Get basis states to consider 
    if N_filling == -1:
        basis_StNrs = np.arange(2**num_of_sites, dtype=np.int64)
    else:
        basis_StNrs = _get_basis_StNr(num_of_sites, N_filling)
    
    # Preallocate arrays
    num_of_states = basis_StNrs.shape[0]
    Es = np.empty((num_of_states,), dtype=np.float64)
    basis_StNrs_filtered = np.empty((num_of_states,), dtype=np.int64)
    num_of_filtered_states = 0
    
    if print_progress:
        print("Collecting all basis states")

    # Loop through all states
    num_of_plaquettes, max_num_of_sites = pl_id_to_site_ids.shape
    for i, StNr in enumerate(basis_StNrs):
        if print_progress:
            print_progress_to_console(i, num_of_states, num_updates=4)

        E = 0
        add_state = True
        # Loop through all plaquettes of the system
        for pl_id in range(num_of_plaquettes):
            n_plaquette = 0
            for j in range(max_num_of_sites):
                site_id = pl_id_to_site_ids[pl_id, j]
                if site_id != -1: # padding guard
                    n_plaquette += _get_n(StNr, site_id, num_of_sites)
            if ground_states_only:
                if n_plaquette != m_ref:
                    add_state = False
                    break
            E += (n_plaquette-m_ref)**2 - m_ref**2

        if add_state:
            basis_StNrs_filtered[num_of_filtered_states] = StNr
            Es[num_of_filtered_states] = E
            num_of_filtered_states += 1
    basis_StNrs = basis_StNrs_filtered[:num_of_filtered_states]
    Es = V*Es[:num_of_filtered_states]
    return Es, basis_StNrs



###################################################################################################
# ROKHSAR KIVELSON HAMILTONIAN
###################################################################################################

def get_RK_ground_state(
        V_RK:float, t_RK:float, m_ref:int, 
        basis_state_StNrs:list[int]|np.ndarray=None, print_progress:bool=False,
        # explicit
        ex_pl_id_to_site_ids:list[list[int]]=None, ex_RK_loops:list[list[int]]=None,
        ex_N_filling:int=None, 
        # object based
        lattice:Lattice=None,
        RK_loops_kwargs:dict=None
) -> tuple[list[HardcoreBosonMBS], dict]:
    r"""
    Compute the ground states of the Rokhsar-Kivelson Hamiltonian:
    
    .. math::

        H = V_RK \sum_{loop} |loop⟩⟨loop| - t_RK \sum_{loop} ( |loop⟩⟨loop_flipped| + h.c. )

    The Hamiltonian is constructed using :func:`get_RK_H` and then diagonalized to obtain the 
    ground-states. Diagnostics from Hamiltonian construction are combined with diagonalization
    information.

    Parameters
    ----------
    See :func:`get_RK_H` for details on parameters.

    Returns
    -------
    ground_states : list[HardcoreBosonMBS]
        List of degenerate ground-states.
    diagnostics : dict
        A dictionary of diagnostics:
        * Diagnostics returned by :func:`get_RK_H`.
        * **walltime_diagonalization** (float): Time spent diagonalizing the Hamiltonian (seconds).
        * **E_ground_state** (float): Ground-state energy.
        * **state_degeneracy** (int): Number of degenerate ground states.
    
    Notes
    -----
    - See :func:`get_RK_H` for details on basis generation and winding sector considerations.
    - The Hamiltonian is diagonalized using either dense (numpy) or sparse (scipy) methods 
    depending on its size. If the number of states is below ``_MAX_SIZE_FOR_DENSE``, a dense 
    diagonalization (``np.linalg.eigh``) is used; otherwise, a sparse solver 
    (``scipy.sparse.linalg.eigsh``) computes the lowest ``_NUM_EIGS_SPARSE`` eigenvalues and 
    eigenvectors. These parameters can be modified in ``main.config.py``.
    """

    # Parameter validation in get_RK_H()
    
    # Get Hamiltonian -----------------------------------------------------------------------------
    H, StNrs, diagnostics_H = get_RK_H(
        V_RK, t_RK, m_ref, basis_state_StNrs, print_progress, ex_pl_id_to_site_ids, ex_RK_loops,
        ex_N_filling, lattice, RK_loops_kwargs)

    # Diagonalize Hamiltonian ---------------------------------------------------------------------
    if print_progress:
        print("Diagonalizing Hamiltonian")
    t0 = time.time()
    if H.shape[0] < _MAX_SIZE_FOR_DENSE:
        H_dense = H.toarray()
        Es, vs = np.linalg.eigh(H_dense)
        idx = np.argsort(Es)
        Es = Es[idx]
        vs = vs[:, idx]
    else:
        Es, vs = eigsh(H, k=_NUM_EIGS_SPARSE, which='SA')
        idx = np.argsort(Es)
        Es = Es[idx]
        vs = vs[:, idx]
    t_diagonalize = time.time()-t0

    # Check degeneracy ----------------------------------------------------------------------------
    E_ground_state = None
    degeneracy = 0
    for E in Es:
        if E_ground_state is None:
            E_ground_state = E
        if np.abs(E_ground_state-E) > _E_TOL_FOR_DEGENERACY:
            break
        degeneracy += 1

    # Construct return objects --------------------------------------------------------------------
    if lattice is not None:
        num_of_sites = lattice.sites.N
    else:
        all_site_ids = [site_id for pl in pl_id_to_site_ids for site_id in pl]
        num_of_sites = max(all_site_ids) + 1

    ground_states:list[HardcoreBosonMBS] = []
    for i in range(degeneracy):
        psi = vs[:, i]
        wave_func = HardcoreBosonMBS.from_StNrs(psi, StNrs, num_of_sites)
        ground_states.append( normalize_state(wave_func, flip_sign_if_negative=True) )
    diagnostics = {
        **diagnostics_H,
        "walltime_diagonalization": float(t_diagonalize),
        "E_ground_state": float(E_ground_state),
        "state_degeneracy": int(degeneracy)}
    return ground_states, diagnostics



def get_RK_H(
        V_RK:float, t_RK:float, m_ref:int, 
        basis_state_StNrs:list[int]|np.ndarray=None, print_progress:bool=False,
        # explicit
        ex_pl_id_to_site_ids:list[list[int]]=None, ex_RK_loops:list[list[int]]=None,
        ex_N_filling:int=None, 
        # object based
        lattice:Lattice=None,
        RK_loops_kwargs:dict=None
) -> tuple[spmatrix, np.ndarray, dict]:
    r"""
    Construct the Rokhsar-Kivelson Hamiltonian for a hardcore boson system on a given lattice.

    The Hamiltonian has the form:

    .. math::

        H = V_RK \sum_{loop} |loop⟩⟨loop| - t_RK \sum_{loop} ( |loop⟩⟨loop_flipped| + h.c. )

    If ``basis_state_StNrs`` is not provided, the function automatically generates all valid
    basis states satisfying the plaquette filling constraint ``m_ref``. For systems with 
    topological sectors (winding sectors), this may produce multiple degenerate states. 
    This can be avoided by providing a pre-filtered basis ``basis_state_StNrs``.

    Parameters
    ----------
    V_RK : float
        Strength of the diagonal RK term.
    t_RK : float
        Amplitude of the off-diagonal RK flipping term.
    m_ref : int
        Filling per plaquette (proportional to effective chemical potential). Note that this 
        parameter is not needed if ``basis_state_StNrs`` is provided.
    basis_state_StNrs : list[int] | np.ndarray, optional
        List of basis states (defined by integers ``StNr`` with ``basis_state = bin(StNr)``).
        If ``None``, all valid basis states will be generated automatically.
    print_progress : bool, default False
        Whether to print information about the progress in the console.

    Choose one of the following:

    **Explicit** ----------------------------------------------------------------------------------

    ex_pl_id_to_site_ids : list[list[int]]
        List of lists, where each sublist contains the site IDs of a plaquette.
        The index corresponds to the plaquette ID.
        Note that total number of sites is determined by the largest site ID in the system.
    ex_RK_loops : list[list[int]]
        List of loops (defined by site IDs) that can be flipped from ``0101...`` to ``1010...`` 
        (if possible) while preserving the overall plaquette filling constraint ``m_ref``.
    ex_N_filling : int
        Number of particles in the system corresponding to the filling per plaquette ``m_ref``.
        Only needed if ``basis_state_StNrs is None``.

    **Object-based** ------------------------------------------------------------------------------

    lattice : Lattice
        The lattice on which the states are defined.
    RK_loops_kwargs : dict, optional
        Keyword arguments forwarded to ``lattice.get_RK_loops(**RK_loops_kwargs)``.

    Returns
    -------
    H : spmatrix
        Sparse CSR matrix representing the RK Hamiltonian in the given basis.
    basis_state_StNrs : np.ndarray
        Array of basis state numbers used to construct ``H``.
    diagnostics : dict
        A dictionary of diagnostics:
        * **walltime_basis_states** (float): Time spent generating/collecting all basis states
        (seconds).
        * **walltime_build_H** (float): Time spent constructing the Hamiltonian (seconds).
        * **number_of_basis_states** (int): Number of basis states of the Hamiltonian.
    """

    # Parameter validation ------------------------------------------------------------------------
    validate_type(V_RK, float, "V_RK")
    validate_type(t_RK, float, "t_RK")
    validate_type(print_progress, bool, "print_progress")
    if basis_state_StNrs is not None:
        if isinstance(basis_state_StNrs, list):
            validate_type(basis_state_StNrs, [list, list, int], "basis_state_StNrs")
            basis_state_StNrs: np.ndarray = np.array(basis_state_StNrs)
        else:
            validate_type(basis_state_StNrs, np.ndarray, "basis_state_StNrs")
            assert basis_state_StNrs.ndim == 1
        m_ref = None
    else:
        validate_type(m_ref, int, "m_ref")
    
    # Determine input mode ------------------------------------------------------------------------
    if lattice is not None:
        validate_type(lattice, Lattice, "lattice")
        input_object_based = True
    else:
        if (ex_N_filling is None or ex_pl_id_to_site_ids is None or ex_RK_loops is None):
            raise ValueError(
                "Provide either the object-based input (`lattice`) "
                "or the explicit input (`ex_pl_id_to_site_ids`, `ex_RK_loops`, `ex_N_filling`). "
                "Both cannot be None.")
        validate_type(ex_N_filling, int, "ex_N_filling")
        validate_type(ex_pl_id_to_site_ids, [list, list, int], "ex_pl_id_to_site_ids")
        validate_type(ex_RK_loops, [list, list, int], "ex_RK_loops")
        input_object_based = False

    # Prepare parameters and converting data structures -------------------------------------------
    if RK_loops_kwargs is None:
        RK_loops_kwargs = {}

    if input_object_based:
        if basis_state_StNrs is None:
            N_filling = lattice.get_N_filling(m_ref)
        num_of_sites = lattice.sites.N
        pl_id_to_site_ids = nested_ListWithShifts_to_padded_array(
            lattice.plaquettes.site_ids, dtype=np.int64)
        RK_loops = lattice.get_RK_loops(**RK_loops_kwargs)
        RK_loops = nested_ListWithShifts_to_padded_array(RK_loops)
    else:
        if basis_state_StNrs is None:
            N_filling = ex_N_filling
        all_site_ids = [site_id for pl in ex_pl_id_to_site_ids for site_id in pl]
        num_of_sites = max(all_site_ids) + 1
        pl_id_to_site_ids = nested_list_to_padded_array(ex_pl_id_to_site_ids, dtype=np.int64)
        RK_loops = nested_list_to_padded_array(ex_RK_loops)

    # Get basis states for Hamiltonian ------------------------------------------------------------
    t0 = time.time()
    if basis_state_StNrs is None:
        _, basis_state_StNrs = _get_all_basis_states(
            num_of_sites, pl_id_to_site_ids, m_ref, N_filling, V=1., ground_states_only=True,
            print_progress=print_progress)
    t_basis_states = time.time()-t0
    
    # Build Hamiltonian ---------------------------------------------------------------------------
    t0 = time.time()
    vals, rows, cols = _get_RK_H_entries(
        V_RK, t_RK, num_of_sites, RK_loops, basis_state_StNrs, print_progress)
    num_of_basis_states = basis_state_StNrs.shape[0]
    H = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(num_of_basis_states, num_of_basis_states)).tocsr()
    assert np.abs(H - H.T).max() < 1e-15
    t_build_H = time.time()-t0
    
    # Construct return objects --------------------------------------------------------------------
    diagnostics = {
        "walltime_basis_states": float(t_basis_states),
        "walltime_build_H": float(t_build_H),
        "number_of_basis_states": int(num_of_basis_states)}
    return H, basis_state_StNrs, diagnostics



@optional_njit()
def _get_RK_H_entries(
        V_RK:float, t_RK:float, num_of_sites:int, RK_loops:np.ndarray, 
        basis_state_StNrs:np.ndarray, print_progress:bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # Precomputing mapping from StNr to index
    StNr_to_index = _get_StNr_to_index(basis_state_StNrs)

    # Find shapes
    num_of_states = basis_state_StNrs.shape[0]
    max_num_of_loops, _ = RK_loops.shape

    # Allocate memory for the entries of the Hamiltonian (value, row, col)
    size_guess = num_of_states * (max_num_of_loops) # diag. x off-diag.
    vals = np.empty(size_guess, dtype=np.float64)
    rows = np.empty(size_guess, dtype=np.int64)
    cols = np.empty(size_guess, dtype=np.int64)
    num_of_vals = 0

    # Build Hamiltonian
    if print_progress:
        print("Building Hamiltonian")
    # go though each state
    for i, StNr in enumerate(basis_state_StNrs):
        if print_progress:
            print_progress_to_console(i, num_of_states, num_updates=4)
        # go though each RK loop
        for loop in RK_loops:
            if _is_loop_flipable(StNr, loop, num_of_sites):
                # diagonal term (|state_with_loop> <state_with_loop|)
                val = V_RK
                vals, rows, cols, num_of_vals = _add_to_H(
                    val, i, i, vals, rows, cols, num_of_vals)
                # off-diagonal term (|state_with_loop> <state_with_loop_flipped|)
                StNr_flipped = _flip_sites_of_StNr(StNr, loop, num_of_sites)
                j_off_diag = StNr_to_index[StNr_flipped]
                val = -t_RK
                vals, rows, cols, num_of_vals = _add_to_H(
                    val, i, j_off_diag, vals, rows, cols, num_of_vals)

    return vals[:num_of_vals], rows[:num_of_vals], cols[:num_of_vals]


@optional_njit()
def _get_StNr_to_index(basis_StNrs):
    if not USE_NUMBA:
        StNr_to_index = {int(StNr): int(i) for i, StNr in enumerate(basis_StNrs)}
    else:
        StNr_to_index = Dict.empty(key_type=types.int64, value_type=types.int64)
        for i in range(len(basis_StNrs)):
            StNr_to_index[basis_StNrs[i]] = i
    return StNr_to_index


@optional_njit()
def _is_loop_flipable(StNr:int, loop:np.ndarray, num_of_sites:int) -> bool:
    n_old = -1
    flipable = True
    for k, site_id in enumerate(loop):
        if site_id != -1: # padding guard
            n = _get_n(StNr, site_id, num_of_sites)
            if n == n_old:
                flipable = False
                break
            n_old = n
    return flipable


@optional_njit(inline="always")
def _add_to_H(
        val:float, row:int, col:int, vals:np.ndarray, rows:np.ndarray, cols:np.ndarray, counter:int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if counter >= vals.size:
        vals, rows, cols = _expand_arrays(vals, rows, cols)
    vals[counter] = val
    rows[counter] = row
    cols[counter] = col
    counter += 1
    return vals, rows, cols, counter


@optional_njit(inline="always")
def _expand_arrays(
    vals:np.ndarray, rows:np.ndarray, cols:np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = vals.size
    new_size = max(1, 2*size)
    vals_new = np.empty(new_size, dtype=vals.dtype)
    rows_new = np.empty(new_size, dtype=rows.dtype)
    cols_new = np.empty(new_size, dtype=cols.dtype)
    vals_new[:size] = vals
    rows_new[:size] = rows
    cols_new[:size] = cols
    return vals_new, rows_new, cols_new
