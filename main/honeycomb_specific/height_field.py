
# General Imports
import numpy as np

# Internal Imports
from main.tools.utils import validate_type
from main.model.lattice import (
    ListWithShifts, LatticeFromUnitCell, nested_ListWithShifts_to_nested_list)
from main.model.unit_cell import HoneycombUnitCell, HoneycombRectangularUnitCell
from main.honeycomb_specific.helpers import occupation_states_to_spin_states


###################################################################################################
# HEIGHT FIELD
###################################################################################################

def get_height_field(
        lattice:LatticeFromUnitCell, state:np.ndarray, m_ref:int, 
        initial_heights:list[int]=[0,0,0], initial_ref_plaquette:int=0):
    validate_type(lattice, LatticeFromUnitCell, "lattice")
    validate_type(state, np.ndarray, "state")
    validate_type(m_ref, int, "m_ref")
    validate_type(initial_heights, [list, int], "initial_heights")
    assert state.ndim == 1
    assert lattice.sites.N == state.shape[0]
    assert m_ref in [1, 2, 3]
    assert (
        isinstance(lattice.unit_cell, HoneycombUnitCell) or 
        isinstance(lattice.unit_cell, HoneycombRectangularUnitCell))
    if isinstance(lattice.unit_cell, HoneycombRectangularUnitCell):
        is_rectangular = True
    else:
        is_rectangular = False
    pl_ref = initial_ref_plaquette
    if is_rectangular:
        initial_pl_ids = [pl_ref+0, pl_ref+2, pl_ref+1]
    else:
        initial_pl_ids = [pl_ref+0, pl_ref+lattice.L1, pl_ref+1]
    heights = np.full((lattice.plaquettes.N), None, dtype=object)
    for part, pl_id0 in enumerate(initial_pl_ids):
        heights[pl_id0] = initial_heights[part]
    heights = assign_heights(lattice, heights, state, m_ref)
    return heights



def assign_heights(
        lattice:LatticeFromUnitCell, heights:np.ndarray, state:np.ndarray, m_ref:int):

    assert (
        isinstance(lattice.unit_cell, HoneycombUnitCell) or 
        isinstance(lattice.unit_cell, HoneycombRectangularUnitCell))

    # Convert the 0,1 to the flux free state
    flux_free_state = occupation_states_to_spin_states(state, m_ref, as_integer=True)

    # prep parameters
    L1, L2 = lattice.L1, lattice.L2
    site_refs = [[0,1],[2,3],[4,5]]
    pls_NN:list[ListWithShifts] = lattice.plaquettes.nearest_neighbors
    pl_site_ids = nested_ListWithShifts_to_nested_list(lattice.plaquettes.site_ids)

    
    # plaquettes of respective partition
    pl_id_to_partition = lattice.unit_cell.get_pl_id_to_partition(lattice.plaquettes.N, L1)
    pl_ids_of_part = [[],[],[]]
    for pl_id in range(lattice.plaquettes.N):
        part = pl_id_to_partition[pl_id]
        pl_ids_of_part[part].append(pl_id)

    # go though each partition to determine the height of a neighboring other partition
    for part in range(3):
        visited = np.zeros(lattice.plaquettes.N, dtype=np.int64)
        
        # go though all plaquettes of the partition and use it as "center" for the flux condition
        # to assign heights
        while np.sum(visited) != lattice.plaquettes.N//3:
            # check if surrounding plaquettes of the other partition have an assigned height
            for pl_id in pl_ids_of_part[part]:
                if visited[pl_id] == 0:
                    j0, h0 = None, None
                    # go through adjacent plaquettes
                    for j, pl_id_NN in enumerate(pls_NN[pl_id].entries):
                        part_adj = pl_id_to_partition[pl_id_NN]
                        if part_adj == (part+2)%3: # only the ones of a certain partition
                            sx = pls_NN[pl_id].x_shifts[j]
                            sy = pls_NN[pl_id].y_shifts[j]
                            if heights[pl_id_NN] is not None:
                                if sx == 0 and sy == 0:
                                    j0 = j
                                    h0 = heights[pl_id_NN]
                                    break
                    # if no height assigned, check next pl_id
                    if j0 is None:
                        continue
                    # otherwise: assign the other two heights (clockwise)
                    for inc in range(1,3):
                        j = (j0 + 2*inc)%6
                        pl_id_NN = pls_NN[pl_id].entries[j]
                        shift_x = pls_NN[pl_id].x_shifts[j]
                        shift_y = pls_NN[pl_id].y_shifts[j]
                        site_ids = np.array(pl_site_ids[pl_id])[site_refs[j//2]]
                        delta_h = np.sum(flux_free_state[site_ids])
                        # since we always determine the height of a certain partition with the same
                        # other partition (e.g. always determine purple with blue) there is no need
                        # to change the "rotation" direction
                        h_NN = h0 + delta_h 
                        if heights[pl_id_NN] is None:
                            if shift_x == 0 and shift_y == 0:
                                heights[pl_id_NN] = h_NN
                        h0 = h_NN
                        # Note that if heights[pl_id_NN] is not None, heights[pl_id_NN] = h_NN is not 
                        # necessarily true since they can differ around the boundary by the winding number
                    visited[pl_id] = 1
    return heights