
# General Imports
import numpy as np

# Internal Imports
from main.tools.utils import validate_type
from main.model.lattice import LatticeFromUnitCell, nested_ListWithShifts_to_nested_list
from main.model.unit_cell import SquareSpinIceUnitCell



###################################################################################################
# HEIGHT FIELD
###################################################################################################

def get_height_field(
        lattice:LatticeFromUnitCell, state:np.ndarray, initial_height:int=0):
    validate_type(lattice, LatticeFromUnitCell, "lattice")
    validate_type(state, np.ndarray, "state")
    validate_type(initial_height, int, "initial_height")
    assert state.ndim == 1
    assert lattice.sites.N == state.shape[0]
    assert isinstance(lattice.unit_cell, SquareSpinIceUnitCell)
    heights = np.full((lattice.plaquettes.N), None, dtype=object)
    # note that the heights do not live on the plaquettes with the 2-in 2-out rule
    heights[0] = initial_height
    heights = assign_heights(lattice, heights, state)
    return heights



def assign_heights(
        lattice:LatticeFromUnitCell, heights:np.ndarray, state:np.ndarray):

    assert isinstance(lattice.unit_cell, SquareSpinIceUnitCell)

    # Convert the 0,1 to the flux free state
    flux_free_state = 2*state-1

    # prep parameters
    L1, L2 = lattice.L1, lattice.L2
    pl_site_ids = nested_ListWithShifts_to_nested_list(lattice.plaquettes.site_ids)

    # plaquettes of respective partition
    pl_ids_of_partA2 = []
    for pl_id in range(lattice.plaquettes.N):
        part = lattice.unit_cell.get_plaquette_partition(pl_id, L1)
        if part == 1:
            pl_ids_of_partA2.append(pl_id)
    # Note that each unit cell has one plaquette of A2

    # Go though all unit cells and assign heights
    visited = np.zeros(lattice.plaquettes.N, dtype=np.int64)
    visited[0] = 1
    for uc_id, pl_idA2 in enumerate(pl_ids_of_partA2):
        site_ids = pl_site_ids[pl_idA2]
        
        # find initial nonzero height
        i0 = None
        h0 = None
        for i in range(4):
            pl_id_B = lattice.unit_cell.get_pl_id_B(uc_id, i, L1, L2)
            if visited[pl_id_B] == 1:
                i0 = i
                h0 = heights[pl_id_B]
                break
        assert i0 is not None

        # assign heights
        for j in range(1,4):
            i = (i0+j)%4
            pl_id_B = lattice.unit_cell.get_pl_id_B(uc_id, i, L1, L2)
            h_new = h0+flux_free_state[site_ids[(i-1)%4]]
            if visited[pl_id_B] != 1:
                heights[pl_id_B] = h_new
                visited[pl_id_B] = 1
            h0 = h_new
    return heights