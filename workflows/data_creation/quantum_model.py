
# General Imports
from pathlib import Path
import numpy as np
import pickle

# Internal Imports
from main import methods, model, honeycomb_specific


###################################################################################################
# CONFIGURATION
###################################################################################################

# Path to the raw data folder
PROCESSED_DATA_PATH = Path("data/processed")



###################################################################################################
# RUN
###################################################################################################

def run(args):
    m_ref = args["m_ref"]
    unit_cell = args["unit_cell"]
    L1 = args["L1"]
    L2 = args["L2"]
    assert (
        isinstance(unit_cell, model.unit_cell.HoneycombRectangularUnitCell) or
        isinstance(unit_cell, model.unit_cell.HoneycombUnitCell))

    # Build lattice
    lattice = model.lattice.LatticeFromUnitCell(L1, L2, unit_cell)
    N_filling = lattice.get_N_filling(m_ref)

    # Get classical ground states as basis states
    states, _ = methods.hamiltonian.get_all_basis_states(
        m_ref, N_filling, lattice=lattice, ground_states_only=True)

    # Filter winding sector
    winding_sectors, sector_collections = honeycomb_specific.sectors.group_states_by_winding_sectors(
        m_ref, states, lattice)

    # RK parameters
    t_RK = 1.
    V_RK_point = t_RK
    RK_point_ind = 100
    V_left  = np.linspace(-5, V_RK_point, RK_point_ind, endpoint=False)
    V_right = np.linspace(V_RK_point, 4, 51)
    V_RKs = np.concatenate([V_left, V_right])
    V_over_t = V_RKs / t_RK

    ground_states = []
    Es = []
    states_flipable = []
    states_staggered = []
    num_of_flipable_loops = []
    num_of_flipable_loops_max = []

    # Build number of flipable loop operator
    for i, basis_states in enumerate(sector_collections):
        basis_state_StNrs = basis_states.as_StNrs()
        list_of_loops = model.lattice.nested_ListWithShifts_to_nested_list(
                lattice.get_RK_loops(m_ref=m_ref))
        occupation_scheme_A = [i%2 for i in range(len(list_of_loops[0]))]
        occupation_scheme_B = [(i+1)%2 for i in range(len(list_of_loops[0]))]
        P_A = model.hilbert_space.projector_matrix(
            list_of_loops, occupation_scheme_A, basis_states.as_array())
        P_B = model.hilbert_space.projector_matrix(
            list_of_loops, occupation_scheme_B, basis_states.as_array())
        n_flipable = P_A+P_B

        # Get RK ground states
        ground_states.append([])
        Es.append([])
        num_of_flipable_loops.append([])
        max_vals = []
        states_flipable.append([])
        states_staggered.append([])
        for j, V_RK in enumerate(V_RKs):
            gs, diagnostics = methods.hamiltonian.get_RK_ground_state(
                V_RK, t_RK, m_ref, basis_state_StNrs=basis_state_StNrs, 
                lattice=lattice, RK_loops_kwargs={"m_ref": m_ref})
            ground_states[i].append(gs)
            Es[i].append(diagnostics["E_ground_state"])
            num_of_flipable_loops[i].append([])
            for psi in gs:
                num_of_flipable_loops[i][j].append(
                    np.real_if_close(model.hilbert_space.expectation_value(n_flipable, psi)))

                if j == 0 or j==len(V_RKs)-1:
                    dominant_basis_states = []
                    psi_amps = psi.amplitudes_as_array()
                    psi_amps_max = np.max(np.abs(psi_amps))
                    basis_state_ids = np.where(np.isclose(np.abs(psi_amps), psi_amps_max))[0]
                    for basis_state_id in basis_state_ids:
                        basis_state = psi.get_basis_state(basis_state_id)
                        dominant_basis_states.append(basis_state)

                if j == 0:
                    states_flipable[i].append(dominant_basis_states)
                elif j == len(V_RKs)-1:
                    states_staggered[i].append(dominant_basis_states)

            max_vals.append(np.max(np.array(num_of_flipable_loops[i][j])))
        num_of_flipable_loops_max.append(max(max_vals))
    sector_max_flipable = num_of_flipable_loops_max.index(max(num_of_flipable_loops_max))

    # Save data
    data = {
        "m_ref": m_ref,
        "unit_cell": unit_cell.__class__.__name__,
        "L1": L1,
        "L2": L2,
        "RK_point_ind": RK_point_ind,
        "V_over_t": V_over_t,
        "winding_sectors": winding_sectors,
        "ground_states": ground_states,
        "Es": Es,
        "num_of_flipable_loops": num_of_flipable_loops,
        "num_of_flipable_loops_max": num_of_flipable_loops_max,
        "sector_max_flipable": sector_max_flipable,
        "states_flipable": states_flipable,
        "states_staggered": states_staggered}
    file_name = f"RK_Model_{unit_cell.__class__.__name__}_{m_ref}m"
    file_path = PROCESSED_DATA_PATH / f"{file_name}.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved as: {file_path}\n")


###################################################################################################
# EXECUTION
###################################################################################################

if __name__ == "__main__":
    unit_cells = [
        model.unit_cell.HoneycombRectangularUnitCell()]
    Ls = [[3, 2]]
    m_refs = [3]
    for i, unit_cell in enumerate(unit_cells):
        L1, L2 = Ls[i]
        for m_ref in m_refs:
            args = {"unit_cell":unit_cell, "m_ref":m_ref, "L1": L1, "L2": L2}
            run(args)