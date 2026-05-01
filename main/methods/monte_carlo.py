
# General Imports
import numpy as np
import time

# Internal Imports
from main.config import _MAX_CONSECUTIVE_DEAD_ENDS, _MAX_CONSECUTIVE_STUCKS
from main.tools.utils import validate_type, ensure_capacity_2D, nested_list_to_padded_array
from main.tools.optional_njit import optional_njit
from main.model.states import HardcoreBosonStateCollection
from main.model.lattice import (
    ListWithShifts, Lattice, get_adjacent_loops_for_local_changes,
    nested_ListWithShifts_to_nested_list)



###################################################################################################
# WRAPPER: STATE SAMPLING (WITHIN SAME ENERGY MANIFOLD)
###################################################################################################

def sample_states(
        initial_state:np.ndarray, num_of_samples:int, num_of_updates_per_sample:int, 
        field_update_method:int, m_ref:int, p_quasi_particle_hop:float, save_states:bool,
        return_as_array:bool=False, worm_site_id_to_bonded_site_id:list[int]=None,
        # explicit
        ex_worm_nearest_neighbors:list[list[int]]=None,
        ex_worm_connecting_site_ids:list[list[list[int]]]=None,
        ex_worm_check_site_ids_of_path:bool=None,
        ex_local_loops:list[list[int]]=None,
        ex_pl_id_to_site_ids:list[list[int]]=None,
        # object based
        lattice:Lattice=None, worm_kwargs:dict=None, local_loop_kwargs:dict=None,
) -> tuple[
    HardcoreBosonStateCollection|np.ndarray, HardcoreBosonStateCollection|np.ndarray|None, dict]:
    r""" 
    Sample new states from an ``initial_state`` within the same energy manifold using the following
    Monte Carlo field update methods (not moving quasi-particles):
    - (0) worm: finds invertible loops (e.g. ``...0101...`` -> ``...1010...``) in a given 
    worm network (defined by ``worm_nearest_neighbors``, ``worm_connecting_site_ids``,
    ``worm_check_site_ids_of_path``). 
    Note that the worm algorithm can create global loops and change the winding 
    sector (if present).
    - (1) local loop: flips ``local_loops`` from ``...0101...`` to ``...1010...``

    and the following quasi-particle update method:
    - quasi-particle hop: change a bond occupation from ``01`` to ``10`` to make a quasi-particle
    hop (only for initial states that already contain quasi-particles). The allowed hops and the
    connectivity of the bonds are determined by the provided worm network.

    **Limitations**: 
    - For the quasi-particle hop update it is assumed that there is a one to one
    mapping between the ``node_id``'s of the worm network and the ``pl_id``'s of the
    ``lattice``. TODO (if needed): introduce a mapping between pl_id and node_id
    - The algorithm currently works only on worm networks that generate loops of even length.
    TODO (if needed): allow the loop to continue
    - For the local loop update only regular loop geometries are supported, where every loop 
    consists of the same number of sites and maintains a constant coordination number with adjacent
    loops.

    Parameters
    ----------
    initial_state : np.ndarray
        Initial configuration used to start the sampling. The total number of particles, the number
        of quasi-particles, and their charges (+ or -) are conserved.
    num_of_samples : int
        Number of new states to generate during the sampling procedure.
    num_of_updates_per_sample : int
        Number of update performed per new state.
    field_update_method : int
        The field update method to be used. ``0`` for the worm update method and ``1`` for the 
        local loop method.
    m_ref : int
        Reference filling per plaquette (proportional to effective chemical potential). 
        Plaquette fillings ``num_of_particles_in_pl`` above ``m_ref`` are considered 
        quasi-particles with the charge ``q=num_of_particles_in_pl-m_ref``.
    p_quasi_particle_hop : float
        Probabilities for selecting the quasi-particle hop method (``0<=p_quasi_particle_hop<=1``).
        Note that ``p_field_update=1-p_quasi_particle_hop``.
    save_states : bool
        Whether to store every state generated during the sampling process or only return the final
        state.
    return_as_array : bool, optional
        If ``True``, return the generated states directly as a NumPy array instead of constructing 
        a ``HardcoreBosonStateCollection`` object.
        This option avoids the temporary RAM spike associated with building the full 
        ``HardcoreBosonStateCollection`` object. Default is ``False``.
    worm_site_id_to_bonded_site_id : list[int], optional
        Mapping of each site ID to its bonded partner site ID, used in the worm algorithm.
        Required for networks with double bonds, i.e., when a node can connect to a neighbor 
        through two single sites. In such cases, each connection must satisfy 
        ``num_of_sites_per_connection == 1`` (see ``ex_worm_connecting_site_ids``).
        
        This mapping is used to prevent artificial worm termination due to immediate backtracking 
        along a double bond. If not provided, the algorithm will fall back to the standard 
        single-bond rules. Default is ``None``.

    Choose one of the following:

    **Explicit** ----------------------------------------------------------------------------------

    ex_worm_nearest_neighbors : list[list[int]]
        Nearest-neighbor node IDs for each node in the network.
        Indexing scheme: ``neighbor_node_id = ex_worm_nearest_neighbors[node_id][connection_id]``.
        The inner list must be ordered consistently with ``ex_worm_connecting_site_ids[node_id]``,
        i.e. both lists must have the same length and matching connection order.
        **Required for**: field update method (0) worm and/or quasi-particle hop
    ex_worm_connecting_site_ids : list[list[list[int]]]
        Site IDs located along the connections between nodes.
        Indexing scheme: ``site_id = ex_worm_connecting_site_ids[node_id][connection_id][k]``
        where: ``node_id`` indexes the node, ``connection_id`` indexes one of its 
        connections, ``k`` runs over the sites forming that connection.
        The number of **sites per connection**, ``num_of_sites_per_connection``,
        **must be constant throughout the entire network** and can be either ``1`` or ``2``.
        If ``num_of_sites_per_connection == 2``, the order of the site IDs is important:
        First entry: site ID associated with the current node (``node_id``).
        Second entry: site ID associated with the neighboring node.
        **Required for**: field update method (0) worm and/or quasi-particle hop
    ex_worm_check_site_ids_of_path : bool
        Whether to verify that no site ID appears more than once in the proposed path to be 
        flipped. This prevents flipping the same site twice, which would invalidate the Monte Carlo
        move.
        Default is ``True``. This is safer but introduces extra computational effort.
        In certain constrained networks (e.g., bipartite spin ice), this check can be disabled 
        (``False``) because the network connectivity guarantees that a loop cannot visit the same 
        site more than once, since the the loop would close beforehand.
        For networks that allow overlapping or self-intersecting loops, this option must remain 
        ``True``.
        **Required for**: field update method (0) worm and/or quasi-particle hop
    ex_local_loops : list[list[int]]
        List of loops (defined by site IDs) that can be flipped if allowed. 
        The site IDs within each loop must be ordered according to their geometric connectivity, 
        i.e. in the cyclic order in which the sites are traversed along the loop (forming a closed 
        path). The order determines the alternating occupation pattern (e.g. ``0101...``<-> 
        ``1010...``) and must therefore reflect the actual circular structure of the loop.
        **Required for**: field update method (1) local loops
    ex_pl_id_to_site_ids : list[list[int]]
        List of lists, where each sublist contains the site IDs of a plaquette.
        The index corresponds to the plaquette ID.
        **Required for**: quasi-particle hop

    **Object-based** ------------------------------------------------------------------------------

    lattice : Lattice
        The lattice on which the states are defined.
    worm_kwargs : dict, optional
        Keyword arguments forwarded to the worm network generator method 
        ``lattice.get_monte_carlo_worm_network(**worm_kwargs)``.
    local_loop_kwargs : dict, optional
        Keyword arguments forwarded to the local loop generator method 
        ``lattice.get_monte_carlo_local_loops(**local_loop_kwargs)``.
 
    Returns
    -------
    states : HardcoreBosonStateCollection | np.ndarray
        The sampled configurations on the lattice.
    quasi_particle_states : HardcoreBosonStateCollection | np.ndarray | None
        The sampled quasi-particle configurations on the dual lattice. Returns ``None`` if no 
        quasi-particles were sampled (e.g., when ``p_quasi_particle_hop == 0`` or the array of QP 
        is all zero). 
        Note that this returns the plaquette occupations not the charge of the quasi-particles.
    diagnostics : dict
        A dictionary of diagnostics:
        * **walltime_compiling** (float): Time spent compiling JIT-compiled functions (seconds).
        * **walltime_sampling** (float): Time spent performing the sampling (seconds).
        * **p_field_empirical** (float): Empirical fraction of updates that used field updates.
        * **p_hop_empirical** (float): Empirical fraction of updates that used quasi-particle hops.
        * **p_field_acceptance** (float): Acceptance rate of field updates, defined as the fraction
        of attempted field updates that were successful.
        * **p_hop_acceptance** (float): Acceptance rate of quasi-particle hop updates, defined as 
        the fraction of attempted hop updates that were successful.
        * **method_blocked_counter** (int): Number of times a move was blocked due to unavailable 
        update options.
        * **worm_loop_histogram** (np.ndarray): A normalized histogram of the occurring loop 
        lengths in the worm algorithm (if method activated).
    """

    # Preparation ---------------------------------------------------------------------------------
    t0 = time.time()
    validate_type(num_of_samples, int, "num_of_samples")
    validate_type(num_of_updates_per_sample, int, "num_of_updates_per_sample")
    validate_type(save_states, bool, "save_states")
    assert num_of_samples >= 0 and num_of_updates_per_sample >= 0

    (node_id_to_neigh_pl_ids, node_id_to_connecting_ids, sites_per_conn, use_extended_checks,
     double_bonds, site_id_to_bonded_site_id, loop_id_to_site_ids, loop_id_to_affected_loop_ids, 
     site_id_to_affected_loop_ids, pl_id_to_site_ids) = _validate_and_prepare_inputs(
        initial_state, field_update_method, m_ref, p_quasi_particle_hop, 
        worm_site_id_to_bonded_site_id,
        ex_pl_id_to_site_ids, ex_worm_nearest_neighbors, ex_worm_connecting_site_ids,
        ex_worm_check_site_ids_of_path, ex_local_loops,
        lattice, worm_kwargs, local_loop_kwargs)
    
    initial_state = np.asarray(initial_state, dtype=np.bool_) # set dtype for initial_state to bool
    t_preparation = time.time()-t0

    # Warm up / compile function (exclude JIT compilation from sampling timing) -------------------
    t0 = time.time()
    _sample_states(
        initial_state, 0, 0,
        field_update_method, p_quasi_particle_hop, save_states,
        node_id_to_neigh_pl_ids, node_id_to_connecting_ids, sites_per_conn, use_extended_checks,
        double_bonds, site_id_to_bonded_site_id,
        loop_id_to_site_ids, loop_id_to_affected_loop_ids, site_id_to_affected_loop_ids,
        m_ref, pl_id_to_site_ids)
    t_compiling = time.time()-t0

    # Sample states with compiled function --------------------------------------------------------
    t0 = time.time()
    states, QP_states, p_empirical, p_acceptance, method_blocked_counter, worm_loop_histogram = (
        _sample_states(
            initial_state, num_of_samples, num_of_updates_per_sample,
            field_update_method, p_quasi_particle_hop, save_states,
            node_id_to_neigh_pl_ids, node_id_to_connecting_ids, sites_per_conn, use_extended_checks,
            double_bonds, site_id_to_bonded_site_id,
            loop_id_to_site_ids, loop_id_to_affected_loop_ids, site_id_to_affected_loop_ids,
            m_ref, pl_id_to_site_ids))
    t_sampling = time.time()-t0

    # Construct return objects --------------------------------------------------------------------
    if not return_as_array:
        states = HardcoreBosonStateCollection(states)
    quasi_particle_states = None
    if QP_states.size != 0:
        if not np.all(QP_states == 0):
            quasi_particle_states = np.abs(QP_states)
            if not return_as_array:
                quasi_particle_states = HardcoreBosonStateCollection(quasi_particle_states)

    worm_loop_histogram = worm_loop_histogram.astype(dtype=np.float64)
    if worm_loop_histogram.sum() != 0:
        worm_loop_histogram /= worm_loop_histogram.sum()
    diagnostics = {
        "walltime_preparation": float(t_preparation),
        "walltime_compiling": float(t_compiling),
        "walltime_sampling": float(t_sampling),
        "p_field_empirical": float(p_empirical[0]),
        "p_hop_empirical": float(p_empirical[1]),
        "p_field_acceptance": float(p_acceptance[0]),
        "p_hop_acceptance": float(p_acceptance[1]),
        "method_blocked_counter": int(method_blocked_counter),
        "worm_loop_histogram": worm_loop_histogram}
    return states, quasi_particle_states, diagnostics



###################################################################################################
# WRAPPER: FIND ALL STATES (WITHIN SAME ENERGY MANIFOLD)
###################################################################################################

def get_all_states_within_manifold(
        initial_state:np.ndarray, field_update_method:int, m_ref:int, p_quasi_particle_hop:float,
        number_of_final_checks:int=1, print_progress:bool=False, return_as_array:bool=False,
        worm_site_id_to_bonded_site_id:list[int]=None,
        # explicit
        ex_worm_nearest_neighbors:list[list[int]]=None,
        ex_worm_connecting_site_ids:list[list[list[int]]]=None,
        ex_worm_check_site_ids_of_path:bool=None,
        ex_local_loops:list[list[int]]=None,
        ex_pl_id_to_site_ids:list[list[int]]=None,
        # object based
        lattice:Lattice=None, worm_kwargs:dict=None, local_loop_kwargs:dict=None,
) -> tuple[HardcoreBosonStateCollection|np.ndarray, dict]:
    r""" 
    Find all degenerate states from a given initial state by repeatedly calling 
    :func:`_sample_states` and removing duplicates. In each iteration, :func:`_sample_states` is
    called with every already discovered state as the initial state to search for new states.

    Note that the number of reachable states and how they are explored depend on the 
    ``field_update_method`` and ``p_quasi_particle_hop``. E.g. local loops do not explore different
    winding sectors (if present), and if ``p_quasi_particle_hop=0`` quasi-particles cannot move.
    
    **Limitations**: This method is bounded by the expected number of states, which determine
    the RAM footprint and the CPU time required for sampling and unique-state filtering.
    However, it is not directly limited by the system size itself.

    Parameters
    ----------
    number_of_final_checks : int
        The number of final verification iterations to perform. A final check is triggered once an
        iteration finds no new states. In a final check, another iteration is executed to attempt
        to find additional states. If no new states are found, the calculation terminates.
    print_progress : bool
        Whether to print the progress of the sampling to the console or not.

    See :func:`_sample_states` for details on the remaining parameters.

    Returns
    -------
    states : HardcoreBosonStateCollection
        All sampled configurations on the lattice.
    diagnostics : dict
        A dictionary of diagnostics:
        * **walltime_preparation** (float): Time spent on one-time setup and precomputations
        (seconds).
        * **walltime_find_states** (float): Total time spent discovering and processing states
        (seconds).
        * **walltime_share_spent_in_filtering** (float): The fraction of "walltime_find_states"
        spent specifically on filtering duplicate configurations.
        * **number_of_iterations** (int): The total number of iterations performed until no new 
        states were found.
    
    See Also
    --------
    _sample_states : The underlying sampling function used to generate states.
    """
    # Preparation ---------------------------------------------------------------------------------
    t0 = time.time()
    validate_type(number_of_final_checks, int, "number_of_final_checks")
    validate_type(print_progress, bool, "print_progress")
    assert number_of_final_checks >= 0

    (node_id_to_neigh_pl_ids, node_id_to_connecting_ids, sites_per_conn, use_extended_checks,
     double_bonds, site_id_to_bonded_site_id, loop_id_to_site_ids, loop_id_to_affected_loop_ids, 
     site_id_to_affected_loop_ids, pl_id_to_site_ids) = _validate_and_prepare_inputs(
        initial_state, field_update_method, m_ref, p_quasi_particle_hop,
        worm_site_id_to_bonded_site_id,
        ex_pl_id_to_site_ids, ex_worm_nearest_neighbors, ex_worm_connecting_site_ids,
        ex_worm_check_site_ids_of_path, ex_local_loops,
        lattice, worm_kwargs, local_loop_kwargs)
    
    initial_state = np.asarray(initial_state, dtype=np.bool_) # set dtype for initial_state to bool

    # Preallocate states
    states = np.zeros((1, initial_state.shape[0]), dtype=np.bool_)
    states[0,:] = initial_state
    num_of_states = 1
    
    t_preparation = time.time()-t0

    # Find all states by sampling and filtering ---------------------------------------------------
    t0 = time.time()
    t_filtering = 0
    final_check_counter = 0
    iteration_counter = 0
    while True:
        if final_check_counter == 0:
            iteration_counter += 1
        num_of_states_old = num_of_states

        # Determine number of new states to be sampled (this can probably be optimized)
        if num_of_states < 500:
            n = 100
        else:
            n = 2

        # Use each found state as initial state and flip n loops
        states, num_of_states = _perform_updates_for_each_state(
            n, states, num_of_states, field_update_method, p_quasi_particle_hop, 
            node_id_to_neigh_pl_ids, node_id_to_connecting_ids, sites_per_conn, 
            use_extended_checks, double_bonds, site_id_to_bonded_site_id, loop_id_to_site_ids, 
            loop_id_to_affected_loop_ids, site_id_to_affected_loop_ids, m_ref, 
            pl_id_to_site_ids)
        
        # Remove duplicates
        t1 = time.time()
        dtype = np.dtype((np.void, states[:num_of_states].dtype.itemsize * states.shape[1]))
        states_view = np.ascontiguousarray(states[:num_of_states]).view(dtype)
        _, idx = np.unique(states_view, return_index=True)
        states = states[idx]
        t_filtering += time.time()-t1

        # Analyze found states
        num_of_states = states.shape[0]
        gained_states = num_of_states - num_of_states_old
        if print_progress:
            if final_check_counter != 0:
                info_string = f"Final Check {final_check_counter}/{number_of_final_checks}: "
            else:
                info_string = f"Iteration {iteration_counter}: "
            print(info_string + f"{num_of_states} total unique states "+
                  f"({gained_states} added in this iteration)")
        if gained_states == 0:
            if final_check_counter == number_of_final_checks:
                break
            else:
                final_check_counter += 1
        else:
            # if the final check gave rise to new states, perform another final check
            if final_check_counter != 0:
                final_check_counter = 0
    t_find_states = time.time()-t0

    # Construct return objects --------------------------------------------------------------------
    if not return_as_array:
        states = HardcoreBosonStateCollection(states)
    diagnostics = {
        "walltime_preparation": float(t_preparation),
        "walltime_find_states": float(t_find_states),
        "walltime_share_spent_in_filtering": float(t_filtering/t_find_states),
        "number_of_iterations": int(iteration_counter)}
    return states, diagnostics



@optional_njit()
def _perform_updates_for_each_state(
    n:int, states:np.ndarray, num_of_states:int,
    field_update_method:int, p_quasi_particle_hop:float,
    # \/ --- for worm --- \/
    node_id_to_neigh_pl_ids:np.ndarray, node_id_to_connecting_ids:np.ndarray, sites_per_conn:int,
    use_extended_checks:bool, double_bonds:bool, site_id_to_bonded_site_id:np.ndarray,
    # \/ --- for local loops --- \/
    loop_id_to_site_ids:np.ndarray, loop_id_to_affected_loop_ids:np.ndarray,
    site_id_to_affected_loop_ids:np.ndarray,
    # \/ --- for quasi-particle hop --- \/
    m_ref:int, pl_id_to_site_ids:np.ndarray
):
    # Check if array needs to be expanded
    needed_storage_size = num_of_states+num_of_states*n
    states = ensure_capacity_2D(states, num_of_states, needed_storage_size, needed_storage_size)

    # Go through each existing state
    start_num_of_states = num_of_states
    for i_state in range(start_num_of_states):
        # for each state perform n updates
        state = states[i_state,:]
        states_new, _, _, _, _, _ = _sample_states(
            state, n, 1,
            field_update_method, p_quasi_particle_hop, True,
            node_id_to_neigh_pl_ids, node_id_to_connecting_ids, sites_per_conn, use_extended_checks,
            double_bonds, site_id_to_bonded_site_id,
            loop_id_to_site_ids, loop_id_to_affected_loop_ids, site_id_to_affected_loop_ids,
            m_ref, pl_id_to_site_ids)
        states[num_of_states:num_of_states+n, :] = states_new
        num_of_states += n
    return states, num_of_states



###################################################################################################
# STATE SAMPLING ROUTINE (WITHIN SAME ENERGY MANIFOLD)
###################################################################################################

@optional_njit()
def _sample_states(
    initial_state:np.ndarray, num_of_samples:int, num_of_updates_per_sample:int, 
    field_update_method:int, p_quasi_particle_hop:float, save_states:bool,
    # \/ --- for worm --- \/
    node_id_to_neigh_pl_ids:np.ndarray, node_id_to_connecting_ids:np.ndarray, sites_per_conn:int,
    use_extended_checks:bool, double_bonds:bool, site_id_to_bonded_site_id:np.ndarray,
    # \/ --- for local loops --- \/
    loop_id_to_site_ids:np.ndarray, loop_id_to_affected_loop_ids:np.ndarray,
    site_id_to_affected_loop_ids:np.ndarray,
    # \/ --- for quasi-particle hop --- \/
    m_ref:int, pl_id_to_site_ids:np.ndarray
):
    # Note: To minimize function call overhead and maximize execution speed, this routine is
    # implemented as a single block rather than being decomposed into subfunctions.
    
    # Find shapes
    num_of_sites = initial_state.shape[0]
    num_of_nodes = node_id_to_neigh_pl_ids.shape[0]
    num_of_neighs_per_node = node_id_to_connecting_ids.shape[1]
    num_of_loops, num_of_site_ids_in_loop = loop_id_to_site_ids.shape
    num_of_affected_loop_ids_for_loops = loop_id_to_affected_loop_ids.shape[1]
    num_of_affected_loop_ids_for_sites = site_id_to_affected_loop_ids.shape[1]
    num_of_plaquettes = pl_id_to_site_ids.shape[0]
    
    # Methods
    p_field_update = 1-p_quasi_particle_hop

    # Allocate states memory array
    if save_states:
        states = np.zeros((num_of_samples, num_of_sites), dtype=np.bool_)
    else:
        states = np.zeros((1, num_of_sites), dtype=np.bool_)
    
    # Allocate quasi-particle states memory array
    if p_quasi_particle_hop != 0:
        if save_states:
            QP_states = np.zeros((num_of_samples, num_of_plaquettes), dtype=np.int8)
        else:
            QP_states = np.zeros((1, num_of_plaquettes), dtype=np.int8)
    else:
        QP_states = np.empty((0, num_of_plaquettes), dtype=np.int8)

    # Initial quasi-particle-state
    num_of_QP = 0
    QP_id_to_pl_id_temp = np.zeros((num_of_plaquettes,), dtype=np.int64)
    initial_QP_state = np.zeros((num_of_plaquettes,), dtype=np.int64)
    if p_quasi_particle_hop != 0:
        for pl_id in range(num_of_plaquettes):
            site_ids = pl_id_to_site_ids[pl_id]
            valid_ids = site_ids[site_ids != -1] # padding guard
            q = np.sum(initial_state[valid_ids]) - m_ref
            initial_QP_state[pl_id] = q
            if q != 0:
                QP_id_to_pl_id_temp[num_of_QP] = pl_id
                num_of_QP += 1
    
    # If no quasi-particles in system => only field updates
    if num_of_QP == 0:
        p_field_update = 1.

    # Allocate worm bookkeeping arrays
    possible_i_neighs = np.zeros((num_of_neighs_per_node,), dtype=np.int64)
    flip_path = np.zeros((num_of_sites,), dtype=np.int64)
    site_id_to_on_path = np.zeros((num_of_sites,), dtype=np.bool_)
    visited_list = np.zeros((num_of_nodes,), dtype=np.int64)
    visited_node_ids = -np.ones((num_of_nodes,), dtype=np.int64)

    # Allocate local loops bookkeeping arrays
    flipable_loop_ids = np.zeros((num_of_loops,), dtype=np.int64)
    loop_id_to_flipable_id = -np.ones((num_of_loops,), dtype=np.int64)
    num_of_flipable_loops = 0

    # Allocate quasi-particle hop bookkeeping arrays
    QP_id_to_pl_id = QP_id_to_pl_id_temp[:num_of_QP].copy() # size=(num_of_QP,), dtype=np.int64

    # Initiate flipable_loop_ids and loop_id_to_flipable_id bookkeeping
    if field_update_method == 1:
        for loop_id in range(num_of_loops):
            num_of_flipable_loops =_update_loop_bookkeeping(
                initial_state, loop_id, loop_id_to_site_ids, 
                flipable_loop_ids, loop_id_to_flipable_id, num_of_flipable_loops,
                num_of_site_ids_in_loop)
        assert num_of_flipable_loops != 0

    # Initiate running variables (modified in place)
    state = initial_state.copy()
    QP_state = initial_QP_state.copy()

    # Initialize sampling diagnostic variables
    field_counter = 0
    hop_counter = 0
    method_blocked_counter = 0
    field_call_counter = 0
    hop_call_counter = 0
    worm_loop_histogram = np.zeros((num_of_sites,), dtype=np.int64)
    
    # SAMPLING ====================================================================================
    for s in range(num_of_samples):

        # ONE SAMPLE ==============================================================================
        rands = np.random.random(num_of_updates_per_sample)
        for u in range(num_of_updates_per_sample):

            # ONE SINGLE UPDATE ===================================================================

            # Choose method
            field_update = (rands[u] < p_field_update)

            # Check if local loops are blocked (if in use)
            if field_update_method == 1: # local loops
                if num_of_flipable_loops == 0:
                    if p_quasi_particle_hop != 0:
                        # only hopping until loops can form again
                        method_blocked_counter += 1
                        field_update = False # hop
            
            # Start update
            dead_end = True
            dead_end_counter = 0
            stuck_in_a_row_count = 0
            while dead_end:
                dead_end = False

                if dead_end_counter == _MAX_CONSECUTIVE_DEAD_ENDS:
                    stuck = True
                    method_blocked_counter += 1
                    # Check if method is stuck (due to field+hop usage) and resolve it if possible
                    if p_field_update != 0 and p_quasi_particle_hop != 0:
                        if field_update:
                            if field_update_method == 0:
                                # worm loops are blocked: only hopping until loops can form again
                                field_update = False # hop
                                stuck = False
                        else: # quasi-particle hop update
                            # hops are blocked: only field updates until the QP can hop again
                            field_update = True
                            stuck = False

                    if stuck:
                        raise ValueError(
                            "_sample_states(): exceeded maximum number of consecutive dead ends "
                            f"({_MAX_CONSECUTIVE_DEAD_ENDS} in a row) while attempting to "
                            "update the state.")
                    else: # Try to resolve stuck state
                        stuck_in_a_row_count += 1
                        dead_end_counter = 0 

                if stuck_in_a_row_count == _MAX_CONSECUTIVE_STUCKS:
                    raise ValueError(
                            "_sample_states(): exceeded maximum number of consecutive stuck states "
                            f"({_MAX_CONSECUTIVE_STUCKS} in a row) while attempting to "
                            "update the state.")

                if field_update:
                    field_call_counter += 1

                    if field_update_method == 0:
                        # =========================================================================
                        # Worm update
                        # =========================================================================

                        # Initial node_id and direction
                        node_id = np.random.randint(num_of_nodes)
                        direction_positive = (np.random.randint(2)==1)
                        visited_count = -1

                        while True:
                            # Log node_id as visited
                            visited_count += 1
                            visited_node_ids[node_id] = visited_count
                            visited_list[visited_count] = node_id

                            # Check options
                            num_of_possible_moves = 0
                            for i_neigh in range(num_of_neighs_per_node):
                                if use_extended_checks:
                                    possible = _is_connection_possible_extended(
                                        direction_positive, node_id, i_neigh, sites_per_conn, 
                                        node_id_to_connecting_ids, site_id_to_on_path, state)
                                else:
                                    possible = _is_connection_possible(
                                        direction_positive, node_id, i_neigh, sites_per_conn, 
                                        node_id_to_connecting_ids, state)
                                    
                                if possible:
                                    possible_i_neighs[num_of_possible_moves] = i_neigh
                                    num_of_possible_moves += 1

                            # If no possible move => dead end
                            if num_of_possible_moves == 0:
                                visited_count += 1
                                dead_end = True
                                break

                            # Choose next node_id
                            i_neigh_chosen = possible_i_neighs[
                                np.random.randint(num_of_possible_moves)]
                            
                            # Add site IDs to flip path
                            site_id_A = node_id_to_connecting_ids[node_id, i_neigh_chosen, 0]
                            flip_path[sites_per_conn*visited_count] = site_id_A
                            if sites_per_conn == 2:
                                site_id_B = node_id_to_connecting_ids[node_id, i_neigh_chosen, 1]
                                flip_path[sites_per_conn*visited_count+1] = site_id_B
                            
                            # Mark sites as visited
                            if use_extended_checks:
                                site_id_to_on_path[site_id_A] = 1
                                if sites_per_conn == 2:
                                    site_id_to_on_path[site_id_B] = 1
                                if double_bonds:
                                    bonded_site_id = site_id_to_bonded_site_id[site_id_A]
                                    site_id_to_on_path[bonded_site_id] = 1
                            
                            # Switch to new node
                            node_id = node_id_to_neigh_pl_ids[node_id, i_neigh_chosen]

                            # Check if loop closed
                            path_index = visited_node_ids[node_id]
                            if path_index != -1:
                                visited_count += 1
                                # Find the start and end point of the loop
                                loop_start_id = sites_per_conn*path_index
                                loop_end_id = sites_per_conn*(visited_count)
                                worm_loop_length = loop_end_id-loop_start_id
                                # Check if loop has even length
                                if sites_per_conn == 1:
                                    # TODO (if needed): allow the loop to continue
                                    assert worm_loop_length%2==0
                                # Flip the state of all sites on the loop
                                for i in range(loop_start_id, loop_end_id):
                                    site_id = flip_path[i]
                                    state[site_id] = not state[site_id]
                                break
                            
                            # Switch direction if only one connecting site
                            if sites_per_conn == 1:
                                direction_positive = not direction_positive
                            
                        # Reset visited_node_ids
                        for i in range(visited_count):
                            visited_node_ids[visited_list[i]] = -1
                        
                        # Reset site_id_to_on_path
                        if use_extended_checks:
                            loop_end_id = sites_per_conn*(visited_count)
                            for i in range(loop_end_id):
                                site_id = flip_path[i]
                                site_id_to_on_path[site_id] = 0
                                if double_bonds:
                                    bonded_site_id = site_id_to_bonded_site_id[site_id]
                                    site_id_to_on_path[bonded_site_id] = 0
                        
                        # end Worm update ---------------------------------------------------------
                    
                    else:
                        # =========================================================================
                        # Local loop update
                        # =========================================================================

                        # Choose loop_id to be flipped
                        i_flipable = np.random.randint(num_of_flipable_loops)
                        loop_id = flipable_loop_ids[i_flipable]

                        # Flip the state of all sites on the loop
                        for i_site in range(num_of_site_ids_in_loop):
                            site_id = loop_id_to_site_ids[loop_id, i_site]
                            state[site_id] = not state[site_id]

                        # Check if surrounding loops are flipable
                        for i_loop in range(num_of_affected_loop_ids_for_loops):
                            loop_id_to_check = loop_id_to_affected_loop_ids[loop_id, i_loop]
                            num_of_flipable_loops =_update_loop_bookkeeping(
                                state, loop_id_to_check, loop_id_to_site_ids, 
                                flipable_loop_ids, loop_id_to_flipable_id, num_of_flipable_loops,
                                num_of_site_ids_in_loop)

                        # end Local loop update ---------------------------------------------------

                else: # (=no field_update) => quasi-particle hop update
                    hop_call_counter += 1

                    # =============================================================================
                    # Quasi-particle hop update
                    # =============================================================================
                    # Note: node_id and pl_id are assumed to be the same here, which is generally
                    # not true. TODO (if needed): introduce a mapping between pl_id and node_id

                    # Initial pl_id and direction
                    QP_id = np.random.randint(num_of_QP)
                    pl_id = QP_id_to_pl_id[QP_id]
                    direction_positive = (QP_state[pl_id] > 0)

                    # Check options
                    num_of_possible_moves = 0
                    for i_neigh in range(num_of_neighs_per_node):
                        possible = _is_connection_possible(
                            direction_positive, pl_id, i_neigh, sites_per_conn,
                            node_id_to_connecting_ids, state)
                        if possible:
                            pl_id_neigh = node_id_to_neigh_pl_ids[pl_id, i_neigh]
                            if QP_state[pl_id_neigh] == 0:
                                possible_i_neighs[num_of_possible_moves] = i_neigh
                                num_of_possible_moves += 1

                    # If no possible move => dead end
                    if num_of_possible_moves == 0:
                        dead_end = True
                    else:
                        # Choose neighboring pl_id to hop to
                        i_neigh_chosen = possible_i_neighs[
                            np.random.randint(num_of_possible_moves)]
                        pl_id_neigh = node_id_to_neigh_pl_ids[pl_id, i_neigh_chosen]

                        # Change state and quasi-particle state
                        site_id_A = node_id_to_connecting_ids[pl_id, i_neigh_chosen, 0]
                        state[site_id_A] = not state[site_id_A]
                        if sites_per_conn == 2:
                            site_id_B = node_id_to_connecting_ids[pl_id, i_neigh_chosen, 1]
                            state[site_id_B] = not state[site_id_B]
                            QP_state[pl_id_neigh] = QP_state[pl_id]
                        else:
                            QP_state[pl_id_neigh] = -QP_state[pl_id]
                        QP_state[pl_id] = 0
                        QP_id_to_pl_id[QP_id] = pl_id_neigh

                        # Check if surrounding loops are flipable
                        if field_update_method == 1: # local loops
                            for i_loop in range(num_of_affected_loop_ids_for_sites):
                                loop_id_to_check = site_id_to_affected_loop_ids[site_id_A, i_loop]
                                num_of_flipable_loops = _update_loop_bookkeeping(
                                    state, loop_id_to_check, loop_id_to_site_ids, 
                                    flipable_loop_ids, loop_id_to_flipable_id, 
                                    num_of_flipable_loops, num_of_site_ids_in_loop)

                            if sites_per_conn == 2:
                                for i_loop in range(num_of_affected_loop_ids_for_sites):
                                    loop_id_to_check = site_id_to_affected_loop_ids[site_id_B, i_loop]
                                    num_of_flipable_loops = _update_loop_bookkeeping(
                                        state, loop_id_to_check, loop_id_to_site_ids, 
                                        flipable_loop_ids, loop_id_to_flipable_id, 
                                        num_of_flipable_loops, num_of_site_ids_in_loop)
                    
                    # end Quasi-particle hop update -----------------------------------------------
                
                if dead_end:
                    dead_end_counter += 1
            
            # Single update completed
            if field_update:
                field_counter += 1
                if field_update_method == 0:
                    worm_loop_histogram[worm_loop_length] += 1
            else: # quasi-particle hop update 
                hop_counter += 1
            
            # end ONE SINGLE UPDATE ===============================================================

        # Sample completed
        if save_states:
            states[s,:] = state.copy()
            if num_of_QP != 0:
                QP_states[s,:] = QP_state.copy()

        # end ONE SAMPLE ==========================================================================

    # Sampling completed
    if not save_states:
        states[0,:] = state.copy()
        if p_quasi_particle_hop != 0:
            QP_states[0,:] = QP_state.copy()
    
    # end SAMPLING ================================================================================
    
    num_of_updates = max( 1, num_of_samples*num_of_updates_per_sample)
    if num_of_updates != 1:
        assert field_counter+hop_counter == num_of_updates
    p_field_empirical = field_counter/num_of_updates
    p_hop_empirical = hop_counter/num_of_updates
    
    p_field_acceptance, p_hop_acceptance = 0, 0
    if field_call_counter != 0:
        p_field_acceptance = field_counter/field_call_counter
    if hop_call_counter != 0:
        p_hop_acceptance = hop_counter/hop_call_counter
    p_empirical = np.array([p_field_empirical, p_hop_empirical])
    p_acceptance = np.array([p_field_acceptance, p_hop_acceptance])

    return (
        states, QP_states, p_empirical, p_acceptance, method_blocked_counter, worm_loop_histogram)



@optional_njit(inline="always")
def _is_connection_possible(
    direction_positive, node_id, i_neigh, sites_per_conn, node_id_to_connecting_ids, state
) -> bool:
    site_id_A = node_id_to_connecting_ids[node_id, i_neigh, 0]
    if sites_per_conn == 2:
        site_id_B = node_id_to_connecting_ids[node_id, i_neigh, 1]
        if direction_positive:
            possible = ((state[site_id_A] == 1) and (state[site_id_B] == 0)) 
        else:
            possible = ((state[site_id_A] == 0) and (state[site_id_B] == 1))
    else:
        if direction_positive:
            possible = (state[site_id_A] == 1)
        else:
            possible = (state[site_id_A] == 0)
    return possible



@optional_njit(inline="always")
def _is_connection_possible_extended(
    direction_positive, node_id, i_neigh, sites_per_conn, node_id_to_connecting_ids, 
    site_id_to_on_path, state
) -> bool:
    # Same implementation as above, but with additional consistency checks.
    # Defined separately to avoid unnecessary runtime when the controlling parameter is known
    # at compile time.
    site_id_A = node_id_to_connecting_ids[node_id, i_neigh, 0]
    if site_id_A == -1: # padding guard
        return False
    possible = not site_id_to_on_path[site_id_A]
    if possible:
        if sites_per_conn == 2:
            site_id_B = node_id_to_connecting_ids[node_id, i_neigh, 1]
            possible = not site_id_to_on_path[site_id_B]
            if possible:
                if direction_positive:
                    possible = ((state[site_id_A] == 1) and (state[site_id_B] == 0)) 
                else:
                    possible = ((state[site_id_A] == 0) and (state[site_id_B] == 1))
        else:
            if direction_positive:
                possible = (state[site_id_A] == 1)
            else:
                possible = (state[site_id_A] == 0)
    return possible



@optional_njit(inline="always")
def _is_loop_flipable(loop_id, loop_id_to_site_ids, state, num_of_site_ids_in_loop) -> bool:
    previous_occupation = state[loop_id_to_site_ids[loop_id, 0]]
    for k in range(1, num_of_site_ids_in_loop):
        occupation = state[loop_id_to_site_ids[loop_id, k]]
        if occupation == previous_occupation:
            return False
        previous_occupation = occupation
    return True



@optional_njit(inline="always")
def _update_loop_bookkeeping(
    state, loop_id_to_check, loop_id_to_site_ids, 
    flipable_loop_ids, loop_id_to_flipable_id, num_of_flipable_loops,
    num_of_site_ids_in_loop
) -> int:
    j_flipable = loop_id_to_flipable_id[loop_id_to_check]
    is_flipable = _is_loop_flipable(
        loop_id_to_check, loop_id_to_site_ids, state, num_of_site_ids_in_loop)
    if is_flipable:
        if j_flipable == -1: # was previously not flipable => new entry
            flipable_loop_ids[num_of_flipable_loops] = loop_id_to_check
            loop_id_to_flipable_id[loop_id_to_check] = num_of_flipable_loops
            num_of_flipable_loops += 1
        else: # was previously flipable as well => no change
            pass
    else: # not flipable
        if j_flipable == -1: # was previously not flipable => no change
            pass
        else: # was previously flipable => remove entry and fill hole
            last_entry_loop_id = flipable_loop_ids[num_of_flipable_loops-1]
            flipable_loop_ids[j_flipable] = last_entry_loop_id
            loop_id_to_flipable_id[last_entry_loop_id] = j_flipable
            loop_id_to_flipable_id[loop_id_to_check] = -1
            num_of_flipable_loops -= 1
    return num_of_flipable_loops



###################################################################################################
# VALIDATE INPUTS AND PREPARE INPUTS
###################################################################################################

def _validate_and_prepare_inputs(
        initial_state:np.ndarray,
        field_update_method:int, m_ref:int, p_quasi_particle_hop:float,
        worm_site_id_to_bonded_site_id:list[int]=None,
        # explicit
        ex_pl_id_to_site_ids:list[list[int]]=None,
        ex_worm_nearest_neighbors:list[list[int]]=None, 
        ex_worm_connecting_site_ids:list[list[list[int]]]=None, 
        ex_worm_check_site_ids_of_path:bool=None,
        ex_local_loops:list[list[int]]=None, 
        # object based
        lattice:Lattice=None, worm_kwargs:dict=None, local_loop_kwargs:dict=None,
):
    # Parameter validation ------------------------------------------------------------------------
    validate_type(initial_state, np.ndarray, "initial_state")
    validate_type(field_update_method, int, "field_update_method")
    validate_type(m_ref, int, "m_ref")
    validate_type(p_quasi_particle_hop, float, "p_quasi_particle_hop")
    if worm_site_id_to_bonded_site_id is not None:
        validate_type(worm_site_id_to_bonded_site_id, [list,int], "worm_site_id_to_bonded_site_id")

    assert np.all((initial_state == 0) | (initial_state == 1)) # check if binary
    assert field_update_method in [0,1]
    assert p_quasi_particle_hop >= 0 and p_quasi_particle_hop <= 1
    num_of_sites = initial_state.shape[0]
    need_worm_network = (field_update_method == 0 or p_quasi_particle_hop != 0)

    # Determine input mode ------------------------------------------------------------------------
    if lattice is not None:
        validate_type(lattice, Lattice, "lattice")
        input_object_based = True
    else:
        worm_inputs_complete = (
            (ex_worm_nearest_neighbors is not None) and (ex_worm_connecting_site_ids is not None) 
            and (ex_worm_check_site_ids_of_path is not None))
        local_loop_inputs_complete = ex_local_loops is not None
        quasi_particle_inputs_complete = ex_pl_id_to_site_ids is not None

        if need_worm_network:
            if not worm_inputs_complete:
                raise ValueError(
                    "Provide the explicit input `ex_worm_nearest_neighbors`, "
                    "`ex_worm_connecting_site_ids`, `ex_worm_check_site_ids_of_path` "
                    "to use field_update_method=0 and/or quasi-particle hop "
                    "(p_quasi_particle_hop != 0). "
                    "Alternatively provide the object-based input (`lattice`).")
            validate_type(ex_worm_nearest_neighbors, [list,list,int], "ex_worm_nearest_neighbors")
            validate_type(
                ex_worm_connecting_site_ids, [list,list,list,int], "ex_worm_connecting_site_ids")
            validate_type(ex_worm_check_site_ids_of_path, bool, "ex_worm_check_site_ids_of_path")
        if field_update_method == 1:
            if not local_loop_inputs_complete:
                raise ValueError(
                    "Provide the explicit input `ex_local_loops` "
                    "to use field_update_method=1. "
                    "Alternatively provide the object-based input (`lattice`).")
            validate_type(ex_local_loops, [list,list,int], "ex_local_loops")
        if p_quasi_particle_hop != 0:
            if not quasi_particle_inputs_complete:
                raise ValueError(
                    "Provide the explicit input `ex_pl_id_to_site_ids`"
                    "to use quasi particle hop (p_quasi_particle_hop != 0). "
                    "Alternatively provide the object-based input (`lattice`).")
            validate_type(ex_pl_id_to_site_ids, [list,list,int], "ex_pl_id_to_site_ids")
        input_object_based = False

    # Prepare parameters --------------------------------------------------------------------------
    if worm_kwargs is None:
        worm_kwargs = {}
    if local_loop_kwargs is None:
        local_loop_kwargs = {}

    if input_object_based:
        if need_worm_network:
            worm_nearest_neighbors, worm_connecting_site_ids, worm_check_site_ids_of_path = (
                lattice.get_monte_carlo_worm_network(**worm_kwargs))
            worm_nearest_neighbors = nested_ListWithShifts_to_nested_list(worm_nearest_neighbors)
            worm_connecting_site_ids = nested_ListWithShifts_to_nested_list(
                worm_connecting_site_ids)
        else:
            worm_nearest_neighbors = None
            worm_connecting_site_ids = None
            worm_check_site_ids_of_path = False
        if field_update_method == 1:
            local_loops = lattice.get_monte_carlo_local_loops(**local_loop_kwargs)
            local_loops = nested_ListWithShifts_to_nested_list(local_loops)
        else:
            local_loops = None
        if p_quasi_particle_hop != 0:
            pl_id_to_site_ids = nested_ListWithShifts_to_nested_list(lattice.plaquettes.site_ids)
        else:
            pl_id_to_site_ids = None
    else:
        worm_nearest_neighbors = ex_worm_nearest_neighbors
        worm_connecting_site_ids = ex_worm_connecting_site_ids
        worm_check_site_ids_of_path = ex_worm_check_site_ids_of_path
        local_loops = ex_local_loops
        pl_id_to_site_ids = ex_pl_id_to_site_ids

    # Validate parameter structure ----------------------------------------------------------------
    if need_worm_network:
        sites_per_conn = _validate_worm_network_connectivity_structure(
            worm_nearest_neighbors, worm_connecting_site_ids)
        assert sites_per_conn in [1,2]
        if worm_site_id_to_bonded_site_id is not None:
            assert sites_per_conn == 1
    if field_update_method == 1:
        assert local_loops is not None
        validate_type(local_loops, [list, list, int], "local_loops")
        num_of_site_ids_in_loop = len(local_loops[0])
        assert num_of_site_ids_in_loop != 0 and num_of_site_ids_in_loop%2==0

    # Precomputing adjacency mappings and converting data structures ------------------------------

    # field update: worm
    if need_worm_network:
        node_id_to_neigh_pl_ids = nested_list_to_padded_array(worm_nearest_neighbors)
        node_id_to_connecting_ids = nested_list_to_padded_array(worm_connecting_site_ids)
        is_padded = (node_id_to_connecting_ids == -1).any()
        if worm_site_id_to_bonded_site_id is not None:
            double_bonds = True
            site_id_to_bonded_site_id = nested_list_to_padded_array(worm_site_id_to_bonded_site_id)
        else:
            double_bonds = False
            site_id_to_bonded_site_id = np.empty((0), dtype=np.int64)
        use_extended_checks = worm_check_site_ids_of_path or is_padded or double_bonds
    else:
        sites_per_conn = 0
        node_id_to_neigh_pl_ids = np.empty((0, 0), dtype=np.int64)
        node_id_to_connecting_ids = np.empty((0, 0, 0), dtype=np.int64)
        use_extended_checks = False
        double_bonds = False
        site_id_to_bonded_site_id = np.empty((0), dtype=np.int64)
    
    # field update: local loops
    if field_update_method == 1:
        loop_id_to_affected_loop_ids = np.array(get_adjacent_loops_for_local_changes(
            num_of_sites, local_loops, local_loops, filter_self=True), dtype=np.int64)
        site_id_to_affected_loop_ids = np.array(get_adjacent_loops_for_local_changes(
            num_of_sites, local_loops, [[i] for i in range(num_of_sites)]), dtype=np.int64)
        loop_id_to_site_ids = np.array(local_loops, dtype=np.int64)
    else:
        loop_id_to_affected_loop_ids = np.empty((0, 0), dtype=np.int64)
        site_id_to_affected_loop_ids = np.empty((0, 0), dtype=np.int64)
        loop_id_to_site_ids = np.empty((0, 0), dtype=np.int64)
    
    # quasi particle hopping
    if p_quasi_particle_hop != 0:
        pl_id_to_site_ids = nested_list_to_padded_array(pl_id_to_site_ids)
    else:
        pl_id_to_site_ids = np.empty((0, 0), dtype=np.int64)
    
    return (
        node_id_to_neigh_pl_ids, node_id_to_connecting_ids, sites_per_conn, use_extended_checks,
        double_bonds, site_id_to_bonded_site_id, loop_id_to_site_ids, loop_id_to_affected_loop_ids,
        site_id_to_affected_loop_ids, pl_id_to_site_ids)



def _validate_worm_network_connectivity_structure(
        worm_nearest_neighbors:list[list[int]]|list[ListWithShifts], 
        worm_connecting_site_ids:list[list[list[int]]]|list[list[ListWithShifts]]
):
    without_neighbor = []
    num_of_sites_per_conn = None
    for node_id, nn_i in enumerate(worm_nearest_neighbors):
        number_of_neighbors = len(nn_i)
        number_of_connections = len(worm_connecting_site_ids[node_id])
        for conn_id, connection in enumerate(worm_connecting_site_ids[node_id]):
            num_of_sites_per_conn_i = len(connection)
            if num_of_sites_per_conn is None:
                num_of_sites_per_conn = num_of_sites_per_conn_i
                if num_of_sites_per_conn not in [1,2]:
                    raise NotImplementedError (
                        f"The number of sites connecting two nodes must be either 1 or 2 "
                        f"for the implemented Monte Carlo algorithms, i.e. "
                        f"len(worm_connecting_site_ids[node_id][conn_id]) in [1,2] but is "
                        f"{num_of_sites_per_conn} for node_id={node_id}, conn_id={conn_id}.")
            if num_of_sites_per_conn_i != num_of_sites_per_conn:
                raise ValueError(
                    f"Length mismatch: the number of sites connecting two nodes must be equal "
                    f"for all connections, i.e. worm_connecting_site_ids[node_id][conn_id]="
                    f"{num_of_sites_per_conn} but is {num_of_sites_per_conn_i} for "
                    f"node_id={node_id}, conn_id={conn_id}.")
        if number_of_neighbors != number_of_connections:
            raise ValueError(
                f"Length mismatch: len(worm_nearest_neighbors[node_id])={number_of_neighbors} "
                f"must be equal to len(worm_connecting_site_ids[node_id])={number_of_connections} "
                f"(where node_id={node_id})")
        if number_of_neighbors == 0:
            without_neighbor.append(str(node_id))
    if len(without_neighbor) != 0:
        raise ValueError(
            "The following node ids have no nearest neighbors:\n"
            + ", ".join(without_neighbor))
    return num_of_sites_per_conn
