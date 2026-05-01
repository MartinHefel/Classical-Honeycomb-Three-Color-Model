
# General Imports
import numpy as np

# Internal Imports
from main.tools.utils import validate_type
from main.model.states import HardcoreBosonStateCollection
from main.model.lattice import ListWithShifts, LatticeFromUnitCell
from main.model.unit_cell import HoneycombUnitCell, HoneycombRectangularUnitCell
from main.honeycomb_specific.helpers import occupation_states_to_spin_states


###################################################################################################
# SUBLATTICE DENSITIES
###################################################################################################

def measure_sublattice_densities(
        states:HardcoreBosonStateCollection|np.ndarray, lattice:LatticeFromUnitCell
):
    validate_type(lattice, LatticeFromUnitCell, "lattice")
    validate_type(lattice.unit_cell, HoneycombUnitCell, "lattice.unit_cell")
    if isinstance(states, HardcoreBosonStateCollection):
        states_array = states.as_array()
    elif isinstance(states, np.ndarray):
        assert states.ndim == 2
        states_array = states
    else:
        validate_type(states, HardcoreBosonStateCollection, "state")
    
    unit_cell:HoneycombUnitCell = lattice.unit_cell
    num_of_sites = lattice.sites.N
    L1 = lattice.L1
    
    # create mask
    sublattice_mask = np.zeros((num_of_sites, 6), dtype=np.bool_) # num_of_sites x num_of_sublt
    for pl_id in range(lattice.plaquettes.N):
        partition = unit_cell.get_plaquette_partition(pl_id, L1)
        if partition == 0:
            site_ids = lattice.plaquettes.site_ids[pl_id].entries
            for sublt_id, site_id in enumerate(site_ids):
                sublattice_mask[site_id, sublt_id] = 1
    
    # get occupation per state and sublattie
    sublattice_occupation = states_array@sublattice_mask # num_of_states x num_of_sublt
    normalization_vector = np.sum(sublattice_occupation, axis=1, keepdims=True)
    #assert np.allclose(np.sum(states_array, axis=1), normalization_vector)

    # average over states
    sublattice_densities = sublattice_occupation / normalization_vector
    return sublattice_densities



###################################################################################################
# WINDING NUMBERS
###################################################################################################

def measure_winding_numbers(
        m_ref:int, states:HardcoreBosonStateCollection|np.ndarray, lattice:LatticeFromUnitCell, 
        check_all:bool=False
) -> np.ndarray:
    validate_type(lattice, LatticeFromUnitCell, "lattice")
    validate_type(check_all, bool, "check_all")
    if isinstance(lattice.unit_cell, HoneycombUnitCell):
        is_rectangular_uc = False
    elif isinstance(lattice.unit_cell, HoneycombRectangularUnitCell):
        is_rectangular_uc = True
    else:
        raise NotImplementedError(
            f"Unit cell {type(lattice.unit_cell).__name__} not supported.")
    if isinstance(states, HardcoreBosonStateCollection):
        num_of_states = states.number_of_states
        states_array = states.as_array()
    elif isinstance(states, np.ndarray):
        assert states.ndim == 2
        num_of_states = states.shape[0]
        states_array = states
    else:
        validate_type(states, HardcoreBosonStateCollection, "state")
    
    L1, L2 = lattice.L1, lattice.L2
    assert L1%3==0, L2%3==0 # to have a valid tripartition
    unit_cell:HoneycombUnitCell = lattice.unit_cell

    # convert to spin state
    spin_state = occupation_states_to_spin_states(states_array, m_ref, as_integer=True)

    # Parameters for the string operators around the torus
    # first entry for string operator in y-direction to measure x-winding
    # second entry for string operator in x-direction to measure y-winding
    if is_rectangular_uc:
        L1p, L2p = L1//3, L2
        pls_per_uc = 2
        uc_site_ids = [
            ListWithShifts([1,3],[0,0],[0,0]),
            ListWithShifts([0,1,1,0],[0,0,1,2],[0,0,0,0])]
        sx_shifts = [0,3]
        sy_shifts = [1,0]
        signs = [ # multiplication factor for the occupation numbers on the string
            [np.array([-1,-1]*L2p), np.array([-1,-1]*L2p), np.array([-1,-1]*L2p)],
            [np.array([-1,-1,1,1]*L1p), np.array([-1,-1,1,1]*L1p), np.array([-1,-1,1,1]*L1p)]]
    else:
        L1p, L2p = L1//3, L2//3
        pls_per_uc = 1
        uc_site_ids = [
            ListWithShifts([0,1,1,0],[0,0,0,0],[0,0,1,2]),
            ListWithShifts([0,1,1,0],[0,0,1,2],[0,0,0,0])]
        sx_shifts = [0,3]
        sy_shifts = [3,0]
        signs = [ # multiplication factor for the occupation numbers on the string
            [np.array([1,1,-1,-1]*L2p), np.array([1,1,-1,-1]*L2p), np.array([1,1,-1,-1]*L2p)],
            [np.array([-1,-1,1,1]*L1p), np.array([-1,-1,1,1]*L1p), np.array([-1,-1,1,1]*L1p)]]

    # Get uc_ids belonging to the partitions
    part_to_uc_ids = [[],[],[]]
    for uc_id in range(lattice.L1*lattice.L2):
        ref_pl_id = uc_id*pls_per_uc
        part = unit_cell.get_plaquette_partition(ref_pl_id, L1)
        part_to_uc_ids[part].append(uc_id)
    assert len(part_to_uc_ids[0]) == len(part_to_uc_ids[1])
    assert len(part_to_uc_ids[1]) == len(part_to_uc_ids[2])

    # Preallocation of windings array
    windings = np.empty((num_of_states, 2, 3))

    # Compute windings
    for dir in range(2): # x- and y-direction (more precisely the 1- and 2- direction)
        global_site_ids = lattice.get_global_patterns(
            uc_site_ids[dir], sx_shifts[dir], sy_shifts[dir])
        for part in range(3): # three (color) partitions
            if check_all:
                number_of_equivalent_loops = len(part_to_uc_ids[part])
            else:
                number_of_equivalent_loops = 1
            for i in range(number_of_equivalent_loops):
                uc_id = part_to_uc_ids[part][i]
                windings_dir_c = np.sum(
                    spin_state[:, global_site_ids[uc_id]]*signs[dir][part], axis=1)
                if i == 0:
                    windings[:, dir, part] = windings_dir_c
                else:
                    assert (np.array_equal(
                        np.round(windings_dir_c, 5), np.round(windings[:, dir, part], 5)))
    return windings