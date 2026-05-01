
# General Imports
import math
import time

# Internal Imports
from main.tools.utils import validate_type, nested_list_to_padded_array
from main.tools.optional_njit import optional_njit
from main.model.lattice import Lattice, nested_ListWithShifts_to_nested_list
from main.methods.hamiltonian import get_all_basis_states



###################################################################################################
# PAULINGS ESTIMATE
###################################################################################################

def get_paulings_estimate_half_filling(sites_per_pl:int, adjacent_pl_per_site:int) -> float:
    validate_type(sites_per_pl, int, "sites_per_pl")
    validate_type(adjacent_pl_per_site, int, "adjacent_pl_per_site")
    num_of_valid_states = math.comb(sites_per_pl, sites_per_pl//2)
    num_of_total_states = 2**sites_per_pl
    P_pl = (num_of_valid_states/num_of_total_states)
    P_valid = P_pl**(adjacent_pl_per_site/sites_per_pl)
    return 2*P_valid



###################################################################################################
# PAULING LIKE ESTIMATE
###################################################################################################

def get_pauling_like_estimate(
        m_ref:int, edge_mappings:list[list[int],list[int]], num_of_cell_plaquettes:int,
        sites_per_pl:int=None, refer_to_sites:bool=True,
        # explicit
        ex_pl_id_to_site_ids_tile:list[list[int]]=None, 
        ex_pl_id_to_site_ids_gap:list[list[int]]=None,
        # object based
        lattice_tile:Lattice=None,
        lattice_gap:Lattice=None,
) -> tuple[float, float]:
    r"""
    Compute a Pauling-like estimate of the number of configurations

    .. math::

        \Omega = ( \alpha_{max} P_{match})^{N_{plaquettes/sites}}

    considering a tile lattice and a gap lattice in between the tiles.

    This method requires a tiling pattern such that all sites are covered by the tile lattice,
    but no two tile lattices share an edge or a site; i.e., there must be a gap between tiles.
    The number of plaquettes in one repeating cell (tiles plus gaps) is given by 
    ``num_of_cell_plaquettes``. The gap lattice may include any number of plaquettes 
    and cover any area within the gaps. The ``edge_mappings`` parameter provides the mapping 
    of site IDs along the gap edges to the corresponding sites on the tile edges.

    The upper bound ``alpha_max`` is computed by relating the number of states that fulfill the
    ``m_ref`` particles-per-plaquette constraint on the tile lattice to a single plaquette or site.

    The tile-gap match probability ``P_match`` is obtained by first computing all tile and gap
    states that fulfill the ``m_ref`` particles-per-plaquette constraint. Then, for each gap 
    state, all possible combinations of tile states are evaluated along the edges to determine 
    which combinations are compatible with that gap state. 
    This procedure effectively checks all combinations of tile edges that satisfy the 
    ``m_ref`` constraint.

    **Limitations**: This method is bounded by the number of possible states ``2**num_of_sites`` 
    of the tile and gap lattice which determines the scaling of both CPU time and RAM usage.

    Parameters
    ----------
    m_ref : int
        Filling per plaquette (proportional to the effective chemical potential).
    edge_mappings : list of [list[int], list[int]]
        List of pairs of site index lists that define the edges connecting the gap and tile 
        lattices.

        - ``len(edge_mappings)`` gives the number of edges.  
        - For each edge, ``edge_mapping[0]`` lists the site IDs on the gap lattice, 
        and ``edge_mapping[1]`` lists the corresponding site IDs on the tile lattice.  
        - The two lists for each edge must have the same length, defining a one-to-one 
        correspondence between the sites along that edge.
    num_of_cell_plaquettes : int
        Number of plaquettes in the cell under consideration.
    sites_per_pl : int, optional
        Number of sites per plaquette. Required if ``refer_to_sites=True``.
    refer_to_sites : bool, default True
        If True, the returned estimate is normalized per site rather than per plaquette.

    Choose one of the following:

    **Explicit** ----------------------------------------------------------------------------------

    ex_pl_id_to_site_ids_tile : list[list[int]], optional
        Explicit list of site IDs for each tile plaquette.
    ex_pl_id_to_site_ids_gap : list[list[int]], optional
        Explicit list of site IDs for each gap plaquette.

    **Object-based** ------------------------------------------------------------------------------

    lattice_tile : Lattice, optional
        Lattice object describing the tile.
    lattice_gap : Lattice, optional
        Lattice object describing the gap.

    Returns
    -------
    alpha_max : float
        Estimated upper bound for the number of states per plaquette (or per site if 
        ``refer_to_sites=True``) computed from the tile lattice.
    P_match : float
        Estimated probability that the gap lattice between the tiles satisfies the 
        ``m_ref`` particles-per-plaquette condition. Value is normalized per plaquette 
        (or per site if ``refer_to_sites=True``).
    diagnostics : dict
        A dictionary of diagnostics:
        * **walltime_find_states** (float): Time spent finding the tile and gap states (seconds).
        * **walltime_compute_P_match** (float): Time spent evaluating tile-gap edge compatibility 
        (seconds).
        * **number_of_tile_states** (int): The number of tile states.
        * **number_of_gap_states** (int): The number of gap states.
    """
    # Parameter validation ------------------------------------------------------------------------
    validate_type(m_ref, int, "m_ref")
    validate_type(edge_mappings, (list,list,int), "edge_mappings")
    validate_type(num_of_cell_plaquettes, int, "num_of_cell_plaquettes")
    validate_type(sites_per_pl, int, "sites_per_pl")
    validate_type(refer_to_sites, bool, "refer_to_sites")
    if refer_to_sites:
        assert sites_per_pl is not None
    for edge_mapping in edge_mappings:
        assert len(edge_mapping) == 2
        assert len(edge_mapping[0]) == len(edge_mapping[1])

    # Prepare parameters and converting data structures -------------------------------------------
    if lattice_tile is not None:
        validate_type(lattice_tile, Lattice, "lattice_tile")
        pl_id_to_site_ids_tile = nested_ListWithShifts_to_nested_list(
            lattice_tile.plaquettes.site_ids)
    else:
        if (ex_pl_id_to_site_ids_tile is None):
            raise ValueError(
                "Provide either the object-based input (`lattice_tile`) "
                "or the explicit input (`ex_pl_id_to_site_ids_tile`). "
                "Both cannot be None.")
        validate_type(ex_pl_id_to_site_ids_tile, [list, list, int], "ex_pl_id_to_site_ids_tile")
        pl_id_to_site_ids_tile = ex_pl_id_to_site_ids_tile
    if lattice_gap is not None:
        validate_type(lattice_gap, Lattice, "lattice_gap")
        pl_id_to_site_ids_gap = nested_ListWithShifts_to_nested_list(
            lattice_gap.plaquettes.site_ids)
    else:
        if (ex_pl_id_to_site_ids_gap is None):
            raise ValueError(
                "Provide either the object-based input (`lattice_gap`) "
                "or the explicit input (`ex_pl_id_to_site_ids_gap`). "
                "Both cannot be None.")
        validate_type(ex_pl_id_to_site_ids_gap, [list, list, int], "ex_pl_id_to_site_ids_gap")
        pl_id_to_site_ids_gap = ex_pl_id_to_site_ids_gap

    edge_mappings = nested_list_to_padded_array(edge_mappings)
    num_of_tile_pls = len(pl_id_to_site_ids_tile)
    num_of_gap_pls = len(pl_id_to_site_ids_gap)
    num_of_gap_pls_in_cell = num_of_cell_plaquettes-num_of_tile_pls

    # Get all ground states -----------------------------------------------------------------------
    t0 = time.time()
    states_tile, _ = get_all_basis_states(
        m_ref, ex_pl_id_to_site_ids=pl_id_to_site_ids_tile, ground_states_only=True)
    states_gap, _ = get_all_basis_states(
        m_ref, ex_pl_id_to_site_ids=pl_id_to_site_ids_gap, ground_states_only=True)
    
    num_of_states_tile = states_tile.number_of_states
    num_of_states_gap = states_gap.number_of_states
    states_tile = states_tile.as_array()
    states_gap = states_gap.as_array()
    t_find_states = time.time()-t0

    # Get match probability -----------------------------------------------------------------------
    t0 = time.time()
    P = get_P_match(states_tile, states_gap, edge_mappings)
    t_compute_P_match = time.time()-t0

    # Compute estimate ----------------------------------------------------------------------------
    # per cell
    alpha_max_per_cell = num_of_states_tile
    P_per_cell = P**(num_of_gap_pls_in_cell/num_of_gap_pls)
    
    # per plaquette
    alpha_max = alpha_max_per_cell**(1/num_of_cell_plaquettes)
    P_match = P_per_cell**(1/num_of_cell_plaquettes)
    
    # per site
    if refer_to_sites:
        alpha_max = alpha_max**(1/sites_per_pl)
        P_match = P_match**(1/sites_per_pl)
    
    diagnostics = {
        "walltime_find_states": float(t_find_states),
        "walltime_compute_P_match": float(t_compute_P_match),
        "number_of_tile_states": int(num_of_states_tile),
        "number_of_gap_states": int(num_of_states_gap)}
    return alpha_max, P_match, diagnostics



@optional_njit()
def get_P_match(states_tile, states_gap, edge_mappings):
    num_gap_states = states_gap.shape[0]
    num_tile_states = states_tile.shape[0]
    num_edges = edge_mappings.shape[0]
    max_edge_len = edge_mappings.shape[2]
    P_match = 0.0
    for i_gap in range(num_gap_states):
        P_state_match = 1.0
        for e in range(num_edges):
            num_matches = 0
            edge_len = 0
            for k in range(max_edge_len):
                if edge_mappings[e,0,k] == -1:
                    break
                edge_len += 1
            for i_tile in range(num_tile_states):
                match = True
                for k in range(edge_len):
                    g = edge_mappings[e,0,k]
                    t = edge_mappings[e,1,k]
                    if states_gap[i_gap, g] != states_tile[i_tile, t]:
                        match = False
                        break
                if match:
                    num_matches += 1
            P_state_match *= num_matches/num_tile_states
        P_match += P_state_match
    
    # Note on normalization:
    # Although it may appear otherwise, the value is correctly normalized without dividing 
    # by `num_gap_states`. The reason is that the tiles cover all sites and no DOF is left over.
    # using `states_gap` is simply a more efficient way of enumerating all edge combinations that 
    # satisfy the ``m_ref`` particles-per-plaquette constraint.
    # This can be verified by setting `ground_states_only=False` for states_gap to allow all gap
    # states. In that case, all combinations are allowed and `P_match=1`.
    return P_match
