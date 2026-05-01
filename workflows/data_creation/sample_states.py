
# General Imports
from pathlib import Path
import pickle
from datetime import datetime

# Internal Imports
from workflows.data_creation.state_factory import get_initial_state_for_MC, place_defects
from main import model, methods



###################################################################################################
# CONFIGURATION
###################################################################################################

RAW_DATA_PATH = Path("data/raw")



###################################################################################################
# RUN
###################################################################################################

def run(args):
    unit_cell = args["unit_cell"]
    L1 = args["L1"]
    L2 = args["L2"]
    m_ref = args["m_ref"]
    defect_mode = args["defect_mode"]
    find_all = args["find_all"]
    if not find_all:
        num_of_samples = args["num_of_samples"]
        num_of_updates_per_sample = L1
    else:
        num_of_samples = None
        num_of_updates_per_sample = None

    if isinstance(unit_cell, model.unit_cell.HoneycombUnitCell):
        assert L1%3==0 and L2%3==0
        print("RAM estimate:", _estimate_RAM_in_GB(L1, L2, m_ref, find_all, num_of_samples))
    elif isinstance(unit_cell, model.unit_cell.HoneycombRectangularUnitCell):
        assert L1%3==0
        print("RAM estimate:", _estimate_RAM_in_GB(L1, 2*L2, m_ref, find_all, num_of_samples))
    else:
        raise TypeError(f"Unsupported unit cell type: {type(unit_cell).__name__}")
    
    assert defect_mode in [None, "doped_with_one", "doped_with_two", "sinlge_pair"]

    

    field_update_method = 0
    if defect_mode is None:
        p_quasi_particle_hop = 1.0
    else:
        p_quasi_particle_hop = 0.4
    run_state_sampling(
        unit_cell, L1, L2, m_ref, defect_mode, find_all, field_update_method, p_quasi_particle_hop,
        num_of_samples, num_of_updates_per_sample, print_progress=True)


###################################################################################################
# STATE SAMPLING
###################################################################################################

def run_state_sampling(
        unit_cell, L1:int, L2:int, m_ref:int, defect_mode:str, find_all:bool,
        field_update_method:int, p_quasi_particle_hop:float,
        num_of_samples:int, num_of_updates_per_sample:int,
        print_progress:bool=False,
):
    start_datetime = datetime.now().strftime("%Y_%m_%d_t%H_%M_%S")

    lattice = model.lattice.LatticeFromUnitCell(L1, L2, unit_cell)
    initial_state = get_initial_state_for_MC(m_ref, lattice)

    if find_all:
        states, diagnostics_state_finding = methods.monte_carlo.get_all_states_within_manifold(
            initial_state, field_update_method, m_ref, p_quasi_particle_hop=p_quasi_particle_hop, 
            number_of_final_checks=2, lattice=lattice, return_as_array=True, 
            print_progress=print_progress)
    else:
        inputs = dict(
            initial_state = initial_state,
            num_of_samples = num_of_samples,
            num_of_updates_per_sample = num_of_updates_per_sample,
            field_update_method = field_update_method,
            m_ref = m_ref,
            p_quasi_particle_hop = p_quasi_particle_hop)
        states, _, diagnostics_state_finding = methods.monte_carlo.sample_states(
                **inputs, save_states=True, lattice=lattice, return_as_array=True)
    number_of_states = states.shape[0]
    if print_progress:
        print(diagnostics_state_finding)
        print(f"Number of states: {number_of_states}")
    
    # Save data -----------------------------------------------------------------------------------
    end_datetime = datetime.now().strftime("%Y_%m_%d_t%H_%M_%S")
    
    inputs = {
        "unit_cell_name": unit_cell.__class__.__name__,
        "L1": L1,
        "L2": L2,
        "m_ref": m_ref,
        "defect_mode": defect_mode,
        "find_all": find_all,
        "field_update_method": field_update_method,
        "p_quasi_particle_hop": p_quasi_particle_hop,
        "num_of_samples": num_of_samples,
        "num_of_updates_per_sample": num_of_updates_per_sample,
        "initial_state": initial_state}
    outputs = {
        "states": states,
        "number_of_states": number_of_states}
    diagnostics = {
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "diagnostics_state_finding": diagnostics_state_finding}
    data = {
        "inputs": inputs,
        "outputs": outputs,
        "diagnostics": diagnostics}

    if find_all:
        find_all_str = "all"
    else:
        find_all_str = "sampled"
    
    if defect_mode is None:
        defect_mode_str = "field"
    else:
        defect_mode_str = defect_mode
        
    file_name = f"States_{unit_cell.__class__.__name__}"
    file_name += f"_{m_ref}m_ref_{L1}L1_{L2}L2_{find_all_str}_{defect_mode_str}"
    file_name += f"_{number_of_states}"

    file_path = RAW_DATA_PATH / f"{file_name}.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    
    if print_progress:
        print("\nSaved as: " + str(file_path))
        print("\nDone")



###################################################################################################
# HELPERS
###################################################################################################

def _estimate_RAM_in_GB(L1, L2, m_ref, find_all, num_of_samples):
    num_sites = 2*L1*L2
    if find_all:
        assert m_ref in [1,2,3]
        alpha = {1:1.05, 2:1.14, 3:1.18}[m_ref]
        expected_num_of_states = alpha**num_sites
    else:
        expected_num_of_states = num_of_samples
    states_array = expected_num_of_states*num_sites # bool

    total_bytes = (states_array)
    return total_bytes/1e9



###################################################################################################
# EXECUTION
###################################################################################################

if __name__ == "__main__":
    # The following shows how the dataset can be generated.
    dry_run = True

    unit_cells = [
        model.unit_cell.HoneycombRectangularUnitCell(),
        model.unit_cell.HoneycombUnitCell()]
    
    uc_m_ref_to_Ls = [
        # HoneycombRectangularUnitCell
        {   "1": [(3,2), (6,3), (9,4), (12,5), (3,3), (6,4)],
            "2": [(3,2), (6,3), (3,3), (6,4)],
            "3": [(3,2), (6,3), (3,3)]},
        # HoneycombUnitCell
        {   "1": [(6,6), (9,9), (12,12)],
            "2": [(3,3), (6,6)],
            "3": [(3,3), (6,6)]}]

    m_refs = [1, 2, 3]
    for uc_id, unit_cell in enumerate(unit_cells):
        for m_ref in m_refs: 
            for L1, L2 in uc_m_ref_to_Ls[uc_id][str(m_ref)]:
                args = {
                    "unit_cell": unit_cell,
                    "L1": L1,
                    "L2": L2,
                    "m_ref": m_ref,
                    "defect_mode": None,
                    "find_all": True,
                    "num_of_samples": None}
                if not dry_run:
                    run(args)
