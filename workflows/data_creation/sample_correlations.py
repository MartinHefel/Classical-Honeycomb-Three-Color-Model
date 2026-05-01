
# General Imports
from pathlib import Path
import numpy as np
import time
import pickle
from datetime import datetime

# Internal Imports
from workflows.data_creation.state_factory import get_initial_state_for_MC, place_defects
from main import model, methods, honeycomb_specific



###################################################################################################
# CONFIGURATION
###################################################################################################

RAW_DATA_PATH = Path("data/raw")



###################################################################################################
# RUN
###################################################################################################

def run(**kwargs):
    run_id = kwargs["run_id"]
    n = kwargs["n"]
    m_ref = kwargs["m_ref"]
    num_of_extra_particles, num_of_excitations = kwargs["defects"]
    partition_to_ignore = kwargs.get("partition_to_ignore", None)
    
    L = 3*2*n
    only_C_r = (L > 60)
    field_update_method = 0
    p_quasi_particle_hop = 0.4
    
    num_of_updates_per_sample = L
    num_of_samples_therm = int(1e6)
    num_of_samples_QP_therm = int(1e6)
    num_of_chunks = 1000
    num_of_samples_per_chunk = int(1e5)
    print("RAM estimate:", _estimate_RAM_in_GB(L, num_of_samples_per_chunk))
    run_measuring_protocol(
            run_id,
            L, m_ref, num_of_extra_particles, num_of_excitations,
            field_update_method, p_quasi_particle_hop,
            num_of_updates_per_sample,
            num_of_samples_therm, num_of_samples_QP_therm,
            num_of_chunks, num_of_samples_per_chunk,
            compute_rho_ip=False, compute_w_idc=False, 
            only_C_r=only_C_r, print_progress=True,
            partition_to_ignore=partition_to_ignore)



###################################################################################################
# MEASURING PROTOCOL
###################################################################################################

def run_measuring_protocol(
        run_id:int,
        L:int, m_ref:int, num_of_extra_particles:int, num_of_excitations:int,
        field_update_method:int, p_quasi_particle_hop:float,
        num_of_updates_per_sample:int,
        num_of_samples_therm:int, num_of_samples_QP_therm:int,
        num_of_chunks:int, num_of_samples_per_chunk:int,
        compute_rho_ip:bool=False, compute_w_idc:bool=False,
        only_C_r:bool=False, print_progress:bool=False,
        partition_to_ignore:int=None,
):

    assert num_of_chunks > 0
    assert num_of_samples_per_chunk > 0
    
    start_datetime = datetime.now().strftime("%Y_%m_%d_t%H_%M_%S")

    # Lattice -------------------------------------------------------------------------------------
    unit_cell = model.unit_cell.HoneycombUnitCell()
    lattice = model.lattice.LatticeFromUnitCell(L1=L, L2=L, unit_cell=unit_cell)

    # Patterns ------------------------------------------------------------------------------------
    if only_C_r:
        patterns_site_ids, k_to_i_field, _, patterns_plaquette_ids, k_to_i_QP, part_to_k_to_i_QP = (
            honeycomb_specific.helpers.get_patterns(lattice))
        patterns_dict = {
            "patterns_site_ids": patterns_site_ids, "k_to_i_field": k_to_i_field, 
            "patterns_plaquette_ids": patterns_plaquette_ids, "k_to_i_QP": k_to_i_QP, 
            "part_to_k_to_i_QP": part_to_k_to_i_QP}
    else:
        patterns_dict = None

    # Initial state -------------------------------------------------------------------------------
    if partition_to_ignore is None:
        initial_state = get_initial_state_for_MC(m_ref, lattice)
    else:
        initial_state = get_initial_state_for_MC(m_ref, lattice, is_for_spin_ice=True)
    
    # Create Spin Ice model (ignore a partition) --------------------------------------------------
    if partition_to_ignore is not None:
        assert not only_C_r 
        # /\ parameters passed to C_ij_to_C_kr in _measure() would need to be adapted (see
        #    process_sampled_correlations.py), to not mix correlations of different sublattices
        #    (unit cell of Honeycomb Spin Ice is larger than the one of full Honeycomb)
        lattice = honeycomb_specific.helpers.to_HoneycombSpinIce(lattice, partition_to_ignore)
        worm_site_id_to_bonded_site_id = lattice.worm_site_id_to_bonded_site_id
    else:
        worm_site_id_to_bonded_site_id = None

    # Thermalization of the field -----------------------------------------------------------------
    t0 = time.time()
    if num_of_samples_therm != 0:
        if print_progress:
            print("\nThermalization of field")
        thermalization_inputs = dict(
            initial_state = initial_state,
            num_of_samples = num_of_samples_therm,
            num_of_updates_per_sample = num_of_updates_per_sample,
            field_update_method = field_update_method,
            m_ref = m_ref,
            p_quasi_particle_hop = 0.)
        therm_state, _, diagnostics_therm = methods.monte_carlo.sample_states(
            **thermalization_inputs, save_states=False, lattice=lattice, return_as_array=True,
            worm_site_id_to_bonded_site_id=worm_site_id_to_bonded_site_id)
        if print_progress:
            print(diagnostics_therm)
        therm_state = therm_state[-1,:].copy()
    else:
        if print_progress:
            print("\nSkipping thermalization of field")
        therm_state = initial_state
        thermalization_inputs = None
        diagnostics_therm = None
    t_thermalization = time.time()-t0

    # Thermalization of quasi-particle ------------------------------------------------------------
    t0 = time.time()
    if (num_of_extra_particles==0 and num_of_excitations==0):
        QP_thermalization_inputs = None
        diagnostics_QP_therm = None
    else:
        if p_quasi_particle_hop == 0:
            raise ValueError(
                "Quasi-particle hopping (`p_quasi_particle_hop`) must be nonzero "
                "when extra particles or excitations are present.")

        if print_progress:
            print(
                f"\nPlacing/removing {num_of_extra_particles} particles and "
                f"creating {num_of_excitations} excitations")
        
        # place defects
        therm_state = place_defects(
            therm_state, num_of_extra_particles, num_of_excitations, m_ref, lattice)

        if num_of_samples_QP_therm!=0:
            if print_progress:
                print("\nThermalization of quasi-particles")
            # quasi particle thermalization
            QP_thermalization_inputs = dict(
                initial_state = therm_state,
                num_of_samples = num_of_samples_QP_therm,
                num_of_updates_per_sample = num_of_updates_per_sample,
                field_update_method = field_update_method,
                m_ref = m_ref,
                p_quasi_particle_hop = p_quasi_particle_hop)
            therm_state, _, diagnostics_QP_therm = methods.monte_carlo.sample_states(
                **QP_thermalization_inputs, save_states=False, lattice=lattice, 
                return_as_array=True, 
                worm_site_id_to_bonded_site_id=worm_site_id_to_bonded_site_id)
            therm_state = therm_state[-1,:].copy()
            if print_progress:
                print(diagnostics_QP_therm)
        else:
            QP_thermalization_inputs = None
            diagnostics_QP_therm = None
            if print_progress:
                print("\nSkipping thermalization of quasi-particles")

    t_QP_thermalization = time.time()-t0

    # Sampling and measuring ----------------------------------------------------------------------
    t0 = time.time()
    if print_progress:
        print("\nSampling and measuring")
    t_measure = 0
    C_ij_field, C_ij_QP, C_r_field, C_r_QP, rho_ip, w_idc = None, None, None, None, None, None
    sampling_diagnostics = None
    sampling_initial_state = therm_state.copy()
    sampling_inputs = dict(
            num_of_samples = num_of_samples_per_chunk,
            num_of_updates_per_sample = num_of_updates_per_sample,
            field_update_method = field_update_method,
            m_ref = m_ref,
            p_quasi_particle_hop = p_quasi_particle_hop)

    for chunk_id in range(num_of_chunks):
        last_run = (chunk_id==num_of_chunks-1)

        # Sample states
        chunk_states, chunk_QP_states, chunk_diagnostics = methods.monte_carlo.sample_states(
            **sampling_inputs, initial_state=therm_state, save_states=True, lattice=lattice, 
            return_as_array=True, worm_site_id_to_bonded_site_id=worm_site_id_to_bonded_site_id)
        
        # Use last state as new initial state
        therm_state = chunk_states[-1,:].copy()

        # Measure
        tm = time.time()
        C_ij_field_c, C_ij_QP_c, C_r_field_c, C_r_QP_c, rho_ip_c, w_idc_c = _measure(
            chunk_states, chunk_QP_states, lattice, compute_rho_ip, compute_w_idc, only_C_r, 
            patterns_dict, num_of_excitations)
        t_measure += time.time()-tm

        # Merge data
        C_ij_field, C_ij_QP, C_r_field, C_r_QP, rho_ip, w_idc = _merge_observables(
            C_ij_field_c, C_ij_QP_c, C_r_field_c, C_r_QP_c, rho_ip_c, w_idc_c, 
            C_ij_field, C_ij_QP, C_r_field, C_r_QP, rho_ip, w_idc,
            last_run, num_of_chunks)
        sampling_diagnostics = _merge_diagnostics(
            chunk_diagnostics, sampling_diagnostics, last_run, num_of_chunks)
        
        # Free up memory
        del chunk_states
        del C_ij_field_c
        del C_ij_QP_c
        if not last_run:
            del chunk_QP_states
        
    last_state = therm_state
    if chunk_QP_states is not None:
        last_QP_state = chunk_QP_states[-1,:].copy()
    else:
        last_QP_state = None

    t_sample_and_measure = time.time()-t0
    sampling_inputs["initial_state"] = sampling_initial_state

    if print_progress:
        print(sampling_diagnostics)
        print(f"t_sample_and_measure={t_sample_and_measure}")
    
    # Protocol completed, save data ---------------------------------------------------------------
    end_datetime = datetime.now().strftime("%Y_%m_%d_t%H_%M_%S")
    
    inputs = {
        "L": L,
        "m_ref": m_ref,
        "num_of_extra_particles": num_of_extra_particles,
        "num_of_excitations": num_of_excitations,
        "thermalization_inputs": thermalization_inputs, 
        "QP_thermalization_inputs": QP_thermalization_inputs,
        "sampling_inputs": sampling_inputs,
        "only_C_r": only_C_r,
        "patterns_dict": patterns_dict,
        "partition_to_ignore": partition_to_ignore}
    outputs = {
        "C_ij_field": C_ij_field,
        "C_ij_QP": C_ij_QP,
        "C_r_field": C_r_field,
        "C_r_QP": C_r_QP,
        "rho_ip": rho_ip,
        "w_idc": w_idc,
        "last_state": last_state,
        "last_QP_state": last_QP_state}
    diagnostics = {
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "t_thermalization": t_thermalization,
        "t_QP_thermalization": t_QP_thermalization,
        "t_sample_and_measure": t_sample_and_measure,
        "diagnostics_therm": diagnostics_therm,
        "diagnostics_QP_therm": diagnostics_QP_therm,
        "sampling_diagnostics": sampling_diagnostics}
    data = {
        "inputs": inputs,
        "outputs": outputs,
        "diagnostics": diagnostics}

    if partition_to_ignore is not None:
        file_subdir = f"L{L}_HonSpinIce/"
    else:
        file_subdir = f"L{L}/"
    file_name = f"{run_id}_MC_SAMPLING"
    file_name += f"_{m_ref}m_ref_{L}L_{num_of_extra_particles}num_of_extra_particles".replace("-", "m")
    file_name += f"_{num_of_excitations}num_of_excitations"
    samples_str = f"{num_of_samples_per_chunk:.0e}".replace("+", "")
    file_name += f"_{num_of_chunks}x{samples_str}samples"
    
    file_path = RAW_DATA_PATH / f"{file_subdir+file_name}.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    
    if print_progress:
        print("\nSaved as: " + str(file_path))
        print("\nDone")



def _measure(
        states: model.states.HardcoreBosonStateCollection, 
        QP_states: model.states.HardcoreBosonStateCollection, 
        lattice:model.lattice.LatticeFromUnitCell,
        compute_rho_ip:bool, compute_w_idc:bool,
        only_C_r:bool=False, patterns_dict:dict=None, num_of_excitations:int=None
):
    if only_C_r:
        assert patterns_dict is not None and num_of_excitations is not None
    C_r_field = None
    C_r_QP = None

    # Two-point correlation function of sites (field): num_of_sites x num_of_sites
    C_ij_field = methods.observables.measure_two_point_correlation_function(
        states, connected=False)
    if only_C_r:
        patterns_site_ids = patterns_dict["patterns_site_ids"]
        k_to_i_field = patterns_dict["k_to_i_field"]
        _, C_kr_field = methods.observables.C_ij_to_C_kr(
            C_ij_field, patterns_site_ids, k_to_i_field)
        C_r_field = np.mean(C_kr_field, axis=0)
        C_ij_field = None

    # Two-point correlation function of QP: num_of_pl x num_of_pl
    if QP_states is not None:
        C_ij_QP = methods.observables.measure_two_point_correlation_function(
            QP_states, connected=False)
        if only_C_r:
            # partition should be the same for each run
            partition = honeycomb_specific.helpers.get_nonzero_partition(lattice, C_ij_QP)
            patterns_plaquette_ids = patterns_dict["patterns_plaquette_ids"]
            if num_of_excitations != 0:
                part_to_k_to_i_QP = patterns_dict["part_to_k_to_i_QP"]
                _, C_kr_QP = methods.observables.C_ij_to_C_kr(
                    C_ij_QP, patterns_plaquette_ids, part_to_k_to_i_QP[partition])
            else:
                k_to_i_QP = patterns_dict["k_to_i_QP"]
                _, C_kr_QP = methods.observables.C_ij_to_C_kr(
                    C_ij_QP, patterns_plaquette_ids, k_to_i_QP)
            C_r_QP = np.mean(C_kr_QP, axis=0)
            C_ij_QP = None
    else:
        C_ij_QP = None
    
    # Sublattice densities with size: num_of_states x 6 (sublattices)
    if compute_rho_ip:
        rho_ip = honeycomb_specific.observables.measure_sublattice_densities(states, lattice)
    else:
        rho_ip = None

    # Winding numbers with size: num_of_states x 2 (directions) x 3 (partitions)
    if compute_w_idc:
        w_idc = honeycomb_specific.observables.measure_winding_numbers(m_ref, states, lattice)
    else:
        w_idc = None
    
    return C_ij_field, C_ij_QP, C_r_field, C_r_QP, rho_ip, w_idc



def _merge_observables(
        C_ij_field_c, C_ij_QP_c, C_r_field_c, C_r_QP_c, rho_ip_c, w_idc_c, 
        C_ij_field, C_ij_QP, C_r_field, C_r_QP, rho_ip, w_idc,
        last_run:bool, num_of_chunks:int
):
    C_ij_field = accumulate(C_ij_field, C_ij_field_c)
    C_ij_QP = accumulate(C_ij_QP, C_ij_QP_c)
    C_r_field = accumulate(C_r_field, C_r_field_c)
    C_r_QP = accumulate(C_r_QP, C_r_QP_c)

    if rho_ip_c is not None:
        if rho_ip is None:
            rho_ip = rho_ip_c
        else:
            rho_ip = np.vstack([rho_ip, rho_ip_c])
    
    if w_idc_c is not None:
        if w_idc is None:
            w_idc = w_idc_c
        else:
            w_idc = np.vstack([w_idc, w_idc_c])
    
    if last_run:
        if C_ij_field is not None:
            C_ij_field /= num_of_chunks
        if C_ij_QP is not None:
            C_ij_QP /= num_of_chunks
        if C_r_field is not None:
            C_r_field /= num_of_chunks
        if C_r_QP is not None:
            C_r_QP /= num_of_chunks
    return C_ij_field, C_ij_QP, C_r_field, C_r_QP, rho_ip, w_idc



def _merge_diagnostics(chunk_diagnostics, diagnostics, last_run:bool, num_of_chunks:int):
    if diagnostics is None:
        diagnostics = {}

    prob_keys = ["p_field_empirical", "p_hop_empirical", "p_field_acceptance", "p_hop_acceptance"]

    for key, val_new in chunk_diagnostics.items():
        if key not in diagnostics or diagnostics[key] is None:
            diagnostics[key] = val_new
            continue
        diagnostics[key] += val_new

    if last_run:
        for key, val_new in chunk_diagnostics.items():
            if key in prob_keys:
                diagnostics[key] /= num_of_chunks
    return diagnostics



###################################################################################################
# HELPERS
###################################################################################################

def accumulate(C, C_c):
    if C is None:
        return C_c
    if C_c is None:
        return C
    C += C_c
    return C



def _estimate_RAM_in_GB(L, num_of_samples_per_chunk, C_ij_computation_chunk_size=int(1e5)):
    # C_ij_computation_chunk_size: chunk size for measure_two_point_correlation_function
    
    num_sites = 2 * L**2
    num_plaquettes = L**2
    field_states_bytes = num_of_samples_per_chunk * num_sites # bool
    QP_states_bytes = num_of_samples_per_chunk * num_plaquettes # int8
    chunk_bytes = C_ij_computation_chunk_size * num_sites * 8
    C_ij_field_bytes = 2*(num_sites**2 * 8)
    C_ij_QP_bytes = 2*(num_plaquettes**2 * 8)

    total_bytes = (
        field_states_bytes+QP_states_bytes+chunk_bytes+C_ij_field_bytes+C_ij_QP_bytes)
    return 2*total_bytes/1e9



###################################################################################################
# EXECUTION
###################################################################################################

if __name__ == "__main__":
    # The following shows how the dataset can be generated.
    # Do not run this sequentially; it is meant for illustration only.

    dry_run = True
    ns = [6, 10, 14, 18]
    num_of_runs = [100, 100, 10, 10]
    m_refs = [1, 2, 3]
    defects = [[0,0], [1,0], [2,0], [0,1]]
    
    for i, n in enumerate(ns):
        for run_id in range(num_of_runs[i]):
            for m_ref in m_refs:
                for defect in defects:
                    args = {
                        "run_id": run_id, 
                        "n": n,
                        "m_ref": m_ref,
                        "defects": defect}
                    if not dry_run:
                        run(**args)
                
                # For L=60 generate Honeycomb Spin Ice data as well
                if n==10:
                    args = {
                        "run_id": run_id, 
                        "n": n,
                        "m_ref": m_ref,
                        "defects": defects[0],
                        "partition_to_ignore": 0}
                    if not dry_run:
                        run(**args)