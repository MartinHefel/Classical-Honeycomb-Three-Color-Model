
# General Imports
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Internal Imports
from main.tools.optional_njit import optional_njit
from main.tools.utils import validate_type, nested_list_to_padded_array
from main.model.lattice import LatticeFromUnitCell
from main.model.unit_cell import HoneycombUnitCell
from main.model.states import HardcoreBosonStateCollection
from main.honeycomb_specific.observables import measure_winding_numbers



###################################################################################################
# WINDING SECTORS
###################################################################################################

def get_sectors_with_wx_wy(
        wx:list[int], wy:list[int], m_ref:int, states:HardcoreBosonStateCollection, 
        lattice:LatticeFromUnitCell, only_connected_sector:bool=False, 
        loops_list:list=None,
) -> list[HardcoreBosonStateCollection]:
    validate_type(wx, [list, int], "wx")
    validate_type(wy, [list, int], "wy")
    validate_type(m_ref, int, "m_ref")
    validate_type(states, HardcoreBosonStateCollection, "state")
    validate_type(lattice, LatticeFromUnitCell, "lattice")
    #validate_type(lattice.unit_cell, HoneycombUnitCell, "lattice.unit_cell")
    assert len(wx)==3
    assert len(wy)==3
    if only_connected_sector:
        if loops_list is None:
            raise ValueError("loops_list must be provided when only_connected_sector=True.")
        validate_type(loops_list, list, "loops_list")
    assert lattice.L1%3==0, lattice.L2%3==0 # to have a valid tripartition
    
    # Winding numbers of states
    windings = measure_winding_numbers(m_ref, states, lattice, check_all=True)
    num_states = windings.shape[0]

    # Filter states with winding number wx wy
    selected_ids = []
    for state_id in range(num_states):
        w_x = windings[state_id, 0, :]
        w_y = windings[state_id, 1, :]
        if np.all(w_x == wx) and np.all(w_y == wy):
            selected_ids.append(state_id)
    if len(selected_ids) == 0:
        return []
    
    filtered_states = HardcoreBosonStateCollection(states.as_array()[selected_ids,:])
    
    if only_connected_sector:
        sector_collections = get_connected_sectors(filtered_states, loops_list)
    else:
        sector_collections = [HardcoreBosonStateCollection(filtered_states.as_array())]

    return sector_collections



def get_minimal_winding_sector(
        m_ref:int, states:HardcoreBosonStateCollection, 
        lattice:LatticeFromUnitCell
) -> tuple[HardcoreBosonStateCollection, np.ndarray]:
    validate_type(m_ref, int, "m_ref")
    validate_type(states, HardcoreBosonStateCollection, "state")
    validate_type(lattice, LatticeFromUnitCell, "lattice")
    unique_windings, sector_collections = group_states_by_winding_sectors(
        m_ref, states, lattice, check_all=True)
    w_sq_min = np.sum(unique_windings[0]**2)
    min_winding_ind = 0
    for winding_id in range(1, unique_windings.shape[0]):
        w_sq = np.sum(unique_windings[winding_id]**2)
        if w_sq < w_sq_min:
            min_winding_ind = winding_id
            w_sq_min = w_sq
    minimal_winding = unique_windings[min_winding_ind]
    return sector_collections[min_winding_ind], minimal_winding



def group_states_by_winding_sectors(
        m_ref:int, states:HardcoreBosonStateCollection, lattice:LatticeFromUnitCell, 
        check_all:bool=False
) -> tuple[np.ndarray, list[HardcoreBosonStateCollection]]:
    validate_type(states, HardcoreBosonStateCollection, "state")
    validate_type(lattice, LatticeFromUnitCell, "lattice")
    
    assert lattice.L1%3==0, lattice.L2%3==0 # to have a valid tripartition
    number_of_states = states.number_of_states

    # measure windings
    windings = measure_winding_numbers(m_ref, states, lattice, check_all)
    
    # group states by winding sector
    flat = np.round( windings.reshape(number_of_states, -1), 5)
    unique_rows, inverse_indices = np.unique(flat, axis=0, return_inverse=True)
    num_unique = unique_rows.shape[0]
    unique_windings = unique_rows.reshape(num_unique, 2, 3)
    groups = [[] for _ in range(num_unique)]

    for state_id, sector_id in enumerate(inverse_indices):
        groups[sector_id].append(state_id)

    # Create a HardcoreBosonStateCollection for each connected sector
    sector_collections = []
    states_array = states.as_array()
    for state_ids in groups:
        sub_states = states_array[state_ids,:]
        sector_collections.append(HardcoreBosonStateCollection(sub_states))

    return unique_windings, sector_collections



###################################################################################################
# CONNECTED SECTORS
###################################################################################################

def get_connected_sectors(
        states: HardcoreBosonStateCollection, loop_list: list
) -> list[HardcoreBosonStateCollection]:
    validate_type(states, HardcoreBosonStateCollection, "state")
    validate_type(loop_list, [list, list, int], "loop_list")

    # Convert parameters to numpy arrays
    states_array = states.as_array(dtype=np.bool_)
    np_loop_list = nested_list_to_padded_array(loop_list)

    # Get connectivity matrix
    connectivity_matrix = get_connectivity_matrix(states_array, np_loop_list)
    assert np.all(connectivity_matrix.T == connectivity_matrix)
    sparse_adj = csr_matrix(connectivity_matrix)

    # Find connected components
    n_comp, labels = connected_components(
        csgraph=sparse_adj,
        directed=False)

    sectors = [[] for _ in range(n_comp)]
    for state_id, comp_id in enumerate(labels):
        sectors[comp_id].append(state_id)

    # Create a HardcoreBosonStateCollection for each connected sector
    sector_collections = []
    states_array = states.as_array()
    for sector in sectors:
        sub_states = states_array[sector,:]
        sector_collections.append(HardcoreBosonStateCollection(sub_states))

    return sector_collections



@optional_njit()
def get_connectivity_matrix(states:np.ndarray, loop_list:np.ndarray):

    num_of_states, num_of_sites = states.shape
    num_of_loops, loop_length = loop_list.shape

    alternating_loop = np.zeros((loop_length), dtype=np.bool_)
    alternating_loop_inv = np.zeros((loop_length), dtype=np.bool_)
    for i in range(loop_length):
        if i%2 == 0:
            alternating_loop[i] = 1
        else:
            alternating_loop_inv[i] = 1

    connectivity_matrix = np.zeros((num_of_states, num_of_states), dtype=np.bool_)

    for state_id in range(num_of_states):
        state = states[state_id, :]
        for loop_id in range(num_of_loops):

            loop = loop_list[loop_id]
    
            # check if loop flipable
            flipable = (np.all(state[loop] == alternating_loop) 
                        or np.all(state[loop] == alternating_loop_inv))

            if flipable:
                # new state obtained by flipping the loop
                flipped_state = state.copy()
                for site_id in loop:
                    assert site_id != -1
                    flipped_state[site_id] = not flipped_state[site_id]

                # find state_id of the flipped state
                for state_id_flipped in range(num_of_states):
                    match = True
                    for i in range(num_of_sites):
                        if flipped_state[i] != states[state_id_flipped, i]:
                            match = False
                            break

                    if match:
                        connectivity_matrix[state_id, state_id_flipped] = 1
                        break
                assert match == True # the other state should always be in the ground state list
    assert np.allclose(connectivity_matrix, connectivity_matrix.T)
    return connectivity_matrix
