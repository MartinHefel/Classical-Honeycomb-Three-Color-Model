
# General Imports
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Internal Imports
from main import model, methods



###################################################################################################
# CONFIGURATION
###################################################################################################

PROCESSED_DATA_PATH = Path("data/processed")



###################################################################################################
# RUN
###################################################################################################

def run(args):
    m_ref = args["m_ref"]
    unit_cell = args["unit_cell"]
    L1 = args["L1"]
    L2 = args["L2"]
    lattice = model.lattice.LatticeFromUnitCell(L1, L2, unit_cell)
    states, diagnostics = methods.hamiltonian.get_all_basis_states(m_ref, V=1/2, lattice=lattice)
    list_of_states_with_same_N = states.split_by_particle_filling()

    N_filling_ref = lattice.get_N_filling(m_ref)

    states = []
    dN_vals = []
    E_vals = []
    degeneracys = []
    nonzero_qs = []
    nonzero_qs_counter = []
    for states_with_same_N in tqdm(list_of_states_with_same_N):
        list_of_states_with_same_N_and_E = states_with_same_N.split_by_energy()
        states.append(list_of_states_with_same_N_and_E)
        for E_indx, states_with_same_N_and_E in enumerate(list_of_states_with_same_N_and_E):
            degeneracy = states_with_same_N_and_E.number_of_states
            E = states_with_same_N_and_E.get_E(0)
            N_filling = None

            nonzero_qs.append([])
            nonzero_qs_counter.append([])
            seen_qs_set = set() # keep track of unique QP configurations as tuples
            for state_id in range(states_with_same_N_and_E.number_of_states):
                state = states_with_same_N_and_E.get_state(state_id)
                if N_filling is None:
                    N_filling = np.sum(state)
                else:
                    assert np.sum(state) == N_filling
                # get quasi-particles
                qs = np.empty((lattice.plaquettes.N), dtype=np.int64)
                for pl_id in range(lattice.plaquettes.N):
                    site_ids = lattice.plaquettes.site_ids[pl_id]
                    qs[pl_id] = np.sum(state[site_ids]) - m_ref
                # get nonzero quasi-particles and sort them (according to charges per plaquette)
                qs_nonzero = qs[qs != 0]
                qs_sorted = np.sort(qs_nonzero)
                # get unique occupation scheme of quasi-partciles
                qs_tuple = tuple(qs_sorted)
                if qs_tuple not in seen_qs_set:
                    seen_qs_set.add(qs_tuple)
                    values, counts = np.unique(qs_nonzero, return_counts=True)
                    order = np.lexsort((values > 0, np.abs(values)))
                    nonzero_qs[-1].append(values[order])
                    nonzero_qs_counter[-1].append(counts[order])

            # Sort unique occupation schemes (first by length, second by number of charges per plaquette)
            if len(nonzero_qs[-1]) != 0:
                sort_indices = sorted(
                    range(len(nonzero_qs[-1])),
                    key=lambda i: (
                        len(nonzero_qs[-1][i]),
                        0 if len(nonzero_qs[-1][i]) == 0 else np.max(np.abs(nonzero_qs[-1][i]))
                    )
                )
                nonzero_qs[-1] = [nonzero_qs[-1][i] for i in sort_indices]
                nonzero_qs_counter[-1] = [nonzero_qs_counter[-1][i] for i in sort_indices]

            dN_vals.append(int(N_filling-N_filling_ref))
            E_vals.append(E)
            degeneracys.append(degeneracy)

    nonzero_qs = [[arr.tolist() for arr in sublist] for sublist in nonzero_qs]
    nonzero_qs_counter = [[arr.tolist() for arr in sublist] for sublist in nonzero_qs_counter]

    end_datetime = datetime.now().strftime("%Y_%m_%d_t%H_%M_%S")
    data = {
        "inputs": {
            "unit_cell_name": unit_cell.__class__.__name__,
            "m_ref":m_ref, "L1":L1, "L2":L2},
        "end_datetime": end_datetime,
        "states": states,
        "dN_vals": dN_vals,
        "E_vals": E_vals,
        "degeneracys": degeneracys,
        "nonzero_qs": nonzero_qs,
        "nonzero_qs_counter": nonzero_qs_counter}

    file_name = f"Spectrum"
    file_name += f"_{unit_cell.__class__.__name__}_{m_ref}m_ref"
    file_name += f"_{L1}L1_{L2}L2"

    file_path = PROCESSED_DATA_PATH / f"{file_name}.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved as: {file_path}\n")



###################################################################################################
# EXECUTION
###################################################################################################

if __name__ == "__main__":
    unit_cells = [
        model.unit_cell.HoneycombUnitCell(), model.unit_cell.HoneycombRectangularUnitCell()]
    Ls = [[3, 3], [3, 2]]
    m_refs = [1,2,3]
    for i, unit_cell in enumerate(unit_cells):
        L1, L2 = Ls[i]
        for m_ref in m_refs:
            args = {"unit_cell":unit_cell, "m_ref":m_ref, "L1": L1, "L2": L2}
            run(args)