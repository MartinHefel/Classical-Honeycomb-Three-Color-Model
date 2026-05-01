
# General Imports
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Internal Imports
from main import methods, model, tools, honeycomb_specific


###################################################################################################
# CONFIGURATION
###################################################################################################

# Path to the raw data folder
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH_BIGSPACE = Path("data_BS/processed")


###################################################################################################
# RUN
###################################################################################################

def run(args):
    num_of_independent_runs = args["num_of_independent_runs"]
    n = args["n"]
    m_refs = args["m_refs"]
    defects = args["defects"]
    is_spin_ice = args.get("is_spin_ice", False)
    check_completeness_only = args.get("check_completeness_only", False)

    data_complete = check_if_files_complete(
        num_of_independent_runs, n, m_refs, defects, is_spin_ice)
    if data_complete:
        print(
            f"{num_of_independent_runs} runs of n={n}, m_refs={m_refs}, defects={defects}, "
            f"is_spin_ice={is_spin_ice}: Data complete")
    if data_complete and not check_completeness_only:
        merge_correlations_data(
            num_of_independent_runs, n, m_refs, defects, is_spin_ice, 
            get_error_for_C_ij_all=False)



###################################################################################################
# MERGE CORRELATIONS DATA
###################################################################################################

def merge_correlations_data(
        num_of_independent_runs, n, m_refs, defects, is_spin_ice, get_error_for_C_ij_all):
    unit_cell = model.unit_cell.HoneycombUnitCell()

    # Build lattice
    L = 6*n
    lattice = model.lattice.LatticeFromUnitCell(L1=L, L2=L, unit_cell=unit_cell)
    num_of_sites = lattice.sites.N
    num_of_pls = lattice.plaquettes.N
    (patterns_site_ids, k_to_i_field, part_to_k_to_i_field,
     patterns_plaquette_ids, k_to_i_QP, part_to_k_to_i_QP) = (
        honeycomb_specific.helpers.get_patterns(lattice))
    site_pattern_length = len(patterns_site_ids[0])
    plaquette_pattern_length = len(patterns_plaquette_ids[0])

    # Go through all parameters
    for m_ref in m_refs:
        for d in range(len(defects)):
            num_of_extra_particles, num_of_excitations = defects[d]

            # Allocate arrays
            if L > 60:
                C_ij_field_all = None
            elif get_error_for_C_ij_all:
                C_ij_field_all = np.empty(
                    (num_of_independent_runs, num_of_sites, num_of_sites), dtype=np.float64)
            else:
                C_ij_field_all = np.empty(
                    (num_of_sites, num_of_sites), dtype=np.float64)
            if not is_spin_ice:
                C_field_r = np.empty(
                    (num_of_independent_runs, site_pattern_length), dtype=np.float64)
            else:
                C_field_r = np.empty(
                    (3, num_of_independent_runs, site_pattern_length), dtype=np.float64)

            if num_of_excitations != 0:
                partition_counts = np.zeros((3,), dtype=np.int64)
                if get_error_for_C_ij_all:
                    C_ij_QP_all = np.zeros(
                        (3, num_of_independent_runs, num_of_pls, num_of_pls), dtype=np.float64)
                else:
                    C_ij_QP_all = np.zeros(
                        (3, num_of_pls, num_of_pls), dtype=np.float64)
            else:
                if get_error_for_C_ij_all:
                    C_ij_QP_all = np.empty(
                        (num_of_independent_runs, num_of_pls, num_of_pls), dtype=np.float64)
                else:
                    C_ij_QP_all = np.empty(
                        (num_of_pls, num_of_pls), dtype=np.float64)
            C_QP_r = np.empty(
                (num_of_independent_runs, plaquette_pattern_length), dtype=np.float64)
            worm_loop_histogram_all = np.empty(
                (num_of_independent_runs, num_of_sites), dtype=np.float64)

            # Fill arrays
            count = 0
            missing_files = []
            for run_id in tqdm( range(num_of_independent_runs), 
                desc=f"m={m_ref} def={defects[d]} | collecting files"
            ):
                file_name = get_file_name(
                    run_id, m_ref, L, num_of_extra_particles, num_of_excitations, is_spin_ice)
                try:
                    data = get_raw_data(file_name)
                except (pickle.UnpicklingError, EOFError):
                    print(f"Skipping corrupted file: {file_name}")
                    continue

                if data is None:
                    missing_files.append(file_name)
                    continue

                # Correlation function of field (living on sites)
                C_ij_field = data["outputs"]["C_ij_field"]
                if C_ij_field is not None and C_ij_field_all is not None:
                    if get_error_for_C_ij_all:
                        C_ij_field_all[count,:,:] = C_ij_field
                    else:
                        C_ij_field_all += C_ij_field
                    if not is_spin_ice:
                        _, C_kr_field = methods.observables.C_ij_to_C_kr(
                            C_ij_field, patterns_site_ids, k_to_i_field)
                        C_field_r[count,:] = np.mean(C_kr_field, axis=0)
                    else:
                        for part in range(3):
                            _, C_kr_field = methods.observables.C_ij_to_C_kr(
                                C_ij_field, patterns_site_ids, part_to_k_to_i_field[part])
                            C_field_r[part, count,:] = np.mean(C_kr_field, axis=0)
                    del C_kr_field
                else:
                    C_field_r[count,:] = data["outputs"]["C_r_field"]

                # Correlation function of QP (living on plaquettes/dual lattice)
                if num_of_excitations != 0 or num_of_extra_particles != 0:
                    C_ij_QP = data["outputs"]["C_ij_QP"]
                    if C_ij_QP is not None:
                        if num_of_excitations != 0:
                            # they only live on one partition, need to filter the partition which is not 0
                            partition = honeycomb_specific.helpers.get_nonzero_partition(lattice, C_ij_QP)
                            part_count = partition_counts[partition]
                            # C_ij_QP_all[partition, count, i, :] = [0,0,..] for i not belonging to partition
                            if get_error_for_C_ij_all:
                                C_ij_QP_all[partition, part_count, :, :] = C_ij_QP
                            else:
                                C_ij_QP_all[partition, :, :] += C_ij_QP
                            _, C_kr_QP = methods.observables.C_ij_to_C_kr(
                                C_ij_QP, patterns_plaquette_ids, part_to_k_to_i_QP[partition])
                            partition_counts[partition] += 1
                        else:
                            if get_error_for_C_ij_all:
                                C_ij_QP_all[count, :, :] = C_ij_QP
                            else:
                                C_ij_QP_all += C_ij_QP
                            _, C_kr_QP = methods.observables.C_ij_to_C_kr(
                                C_ij_QP, patterns_plaquette_ids, k_to_i_QP)
                        C_QP_r[count,:] = np.mean(C_kr_QP, axis=0)
                    else:
                        C_QP_r[count,:] = data["outputs"]["C_r_QP"]

                # Worm loop histogram
                worm_loop_histogram_all[count,:] = (
                    data["diagnostics"]["sampling_diagnostics"]["worm_loop_histogram"])
                
                count += 1
            
            # Missing files
            if len(missing_files) != 0:
                print("Missing files:")
                for file in missing_files:
                    print(file)

            # Average data over runs and determine statistical error
            if count != 0 and C_ij_field_all is not None:
                if get_error_for_C_ij_all:
                    C_ij_field_all, eps_C_ij_field_all = tools.utils.average_over_first_axis(
                        C_ij_field_all[:count,:,:])
                else:
                    C_ij_field_all /= count
                    eps_C_ij_field_all = None
            else:
                C_ij_field_all, eps_C_ij_field_all = None, None
            if num_of_excitations != 0:
                C_ij_QP_temp, eps_C_ij_QP_temp = None, None
                for partition in range(3):
                    part_count = partition_counts[partition]
                    if part_count != 0:
                        if get_error_for_C_ij_all:
                            C_ij_QP_all_part, eps_C_ij_QP_all_part = tools.utils.average_over_first_axis(
                                C_ij_QP_all[partition, :part_count, :, :])
                        else:
                            C_ij_QP_all_part = C_ij_QP_all[partition, :, :]/part_count
                            eps_C_ij_QP_all_part = None
                        if C_ij_QP_temp is None:
                            C_ij_QP_temp = C_ij_QP_all_part
                        else:
                            C_ij_QP_temp += C_ij_QP_all_part
                        if eps_C_ij_QP_temp is None:
                            eps_C_ij_QP_temp = eps_C_ij_QP_all_part
                        else:
                            eps_C_ij_QP_temp += eps_C_ij_QP_all_part
                C_ij_QP_all, eps_C_ij_QP_all = C_ij_QP_temp, eps_C_ij_QP_temp
            else:
                if count != 0:
                    if get_error_for_C_ij_all:
                        C_ij_QP_all, eps_C_ij_QP_all = tools.utils.average_over_first_axis(
                            C_ij_QP_all[:count, :, :])
                    else:
                        C_ij_QP_all /= count
                        eps_C_ij_QP_all = None
                else:
                    C_ij_QP_all, eps_C_ij_QP_all = None, None
            if not is_spin_ice:
                C_field_r, eps_C_field_r = tools.utils.average_over_first_axis(C_field_r[:count,:])
            else:
                C_field_r_list, eps_C_field_r_list = [], []
                for part in range(3):
                    C_field_r_part, eps_C_field_r_part = tools.utils.average_over_first_axis(
                        C_field_r[part,:count,:])
                    C_field_r_list.append(C_field_r_part)
                    eps_C_field_r_list.append(eps_C_field_r_part)
                C_field_r, eps_C_field_r = C_field_r_list, eps_C_field_r_list
            C_QP_r, eps_C_QP_r = tools.utils.average_over_first_axis(C_QP_r[:count,:])
            worm_loop_histogram_all, eps_worm_loop_histogram_all = (
                tools.utils.average_over_first_axis(worm_loop_histogram_all[:count,:]))

            if L > 60:
                patterns_site_ids = None
                patterns_plaquette_ids = None

            # Prepare data
            end_datetime = datetime.now().strftime("%Y_%m_%d_t%H_%M_%S")
            data = {
                "inputs": {
                    "L": L, "m_ref":m_ref, "num_of_extra_particles":num_of_extra_particles,
                    "num_of_excitations":num_of_excitations, "is_spin_ice":is_spin_ice},
                "end_datetime": end_datetime,
                "patterns_site_ids": patterns_site_ids,
                "patterns_plaquette_ids": patterns_plaquette_ids,

                "C_ij_field_all": C_ij_field_all,
                "eps_C_ij_field_all": eps_C_ij_field_all,
                "C_ij_QP_all": C_ij_QP_all,
                "eps_C_ij_QP_all": eps_C_ij_QP_all,

                "C_field_r": C_field_r,
                "eps_C_field_r": eps_C_field_r,
                "C_QP_r": C_QP_r,
                "eps_C_QP_r": eps_C_QP_r,

                "worm_loop_histogram_all": worm_loop_histogram_all,
                "eps_worm_loop_histogram_all": eps_worm_loop_histogram_all}

            # Save data
            file_name = "Correlations"
            if is_spin_ice:
                file_name += "_HonSpinIce"
            if num_of_extra_particles >= 0:
                file_name += f"_{m_ref}m_ref_{L}L_{num_of_extra_particles}num_of_extra_particles"
            else:
                file_name += f"_{m_ref}m_ref_{L}L_m{abs(num_of_extra_particles)}num_of_extra_particles"
            file_name += f"_{num_of_excitations}num_of_excitations"
            
            file_path = PROCESSED_DATA_PATH / f"{file_name}.pkl"

            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            
            print(f"Saved as: {file_path}\n")



###################################################################################################
# STRUCTURE FACTOR
###################################################################################################

def get_structure_factor(
        m_ref:int, L:int, is_spin_ice:bool, u1_lim_full:list[float], u2_lim_full:list[float],
        num_tiles_1:int, num_tiles_2:int, Nq_per_tile:int, tilde_idx:int, tilde_idy:int,
        defect:list[int]=[0,0], check_completeness_only:bool=False):
    num_of_extra_particles, num_of_excitations = defect
    if is_spin_ice:
        file_name = "Correlations_HonSpinIce_"
    else:
        file_name = "Correlations_"
    file_name += f"{m_ref}m_ref_{L}L_{num_of_extra_particles}num_of_extra_particles_"
    file_name += f"{num_of_excitations}num_of_excitations"
    file_path = PROCESSED_DATA_PATH / f"{file_name}.pkl"
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    try:
        inputs = data["inputs"]
        L = inputs["L"]
        m_ref = inputs["m_ref"]
        num_of_extra_particles = inputs["num_of_extra_particles"]
        num_of_excitations = inputs["num_of_excitations"]
        is_spin_ice = inputs["is_spin_ice"]
        C_ij_field_all = data["C_ij_field_all"]
        print(f"{file_path}: is complete")
    except:
        print(f"{file_path}: not complete / corrupted")

    if not check_completeness_only:
        u1_tiles, u2_tiles, _ = generate_tiles(
            u1_lim_full, u2_lim_full, num_tiles_1, num_tiles_2, Nq_per_tile)
        tile_id = tilde_idx*num_tiles_2+tilde_idy

        unit_cell = model.unit_cell.HoneycombUnitCell()
        lattice = model.lattice.LatticeFromUnitCell(L, L, unit_cell)
        C_ij_field_connected = C_ij_field_all-(m_ref/6)**2
        Sq_grid, qx, qy = methods.observables.compute_static_structure_factor_grid(
            C_ij_field_connected, lattice, us_1=u1_tiles[tile_id], us_2=u2_tiles[tile_id])
        if len(u1_tiles) == len(u2_tiles) == 1:
            save_structure_factor(Sq_grid, qx, qy, None, inputs)
        else:
            save_structure_factor(Sq_grid, qx, qy, tile_id, inputs)


def stitch_structure_factor_tiles(
        num_tiles_1, num_tiles_2, Nq_per_tile, m_ref, L, is_spin_ice=False, 
        defect=[0,0]):
    Nq_total_1 = num_tiles_1 * Nq_per_tile
    Nq_total_2 = num_tiles_2 * Nq_per_tile
    Sq_full = np.zeros((Nq_total_1, Nq_total_2))
    qx_full = np.zeros_like(Sq_full)
    qy_full = np.zeros_like(Sq_full)

    save_file = True
    for tile_id in tqdm( range(num_tiles_1*num_tiles_2), 
        desc=f"m={m_ref} is_spin_ice={is_spin_ice} | collecting files"
    ):
        file_name = get_structure_factor_file_name(
            tile_id, m_ref, L, defect[0], defect[1], is_spin_ice)
        
        file_path = PROCESSED_DATA_PATH_BIGSPACE / f"{file_name}.pkl"

        stitch = False
        if not file_path.exists():
            print(f"File not found: tile_id:{tile_id}")
            save_file = False
        else:
            if file_path.stat().st_size == 0:
                print(f"File is empty: tile_id:{tile_id}")
                save_file = False
            else:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    stitch = True

        if stitch:
            inputs = data["inputs"]
            Sq = data["Sq_grid"]
            qx = data["qx"]
            qy = data["qy"]
            tile_id = data["tile_id"]
            i, j = tile_id//num_tiles_2, tile_id % num_tiles_2

            n_i, n_j = Sq.shape
            i0 = i * Nq_per_tile
            j0 = j * Nq_per_tile
            i1 = i0 + n_i
            j1 = j0 + n_j
            Sq_full[i0:i1, j0:j1] = Sq
            qx_full[i0:i1, j0:j1] = qx
            qy_full[i0:i1, j0:j1] = qy
    if True:
        save_structure_factor(Sq_full, qx_full, qy_full, None, inputs)


def generate_tiles(u1_lim, u2_lim, num_tiles_1, num_tiles_2, Nq_per_tile):
    Nq_total_1 = num_tiles_1 * Nq_per_tile
    Nq_total_2 = num_tiles_2 * Nq_per_tile

    u1_full = np.linspace(u1_lim[0], u1_lim[1], Nq_total_1, endpoint=True)
    u2_full = np.linspace(u2_lim[0], u2_lim[1], Nq_total_2, endpoint=True)

    u1s = []
    u2s = []
    tile_ids = []
    for i in range(num_tiles_1):
        for j in range(num_tiles_2):
            i0 = i * Nq_per_tile
            i1 = (i + 1) * Nq_per_tile
            j0 = j * Nq_per_tile
            j1 = (j + 1) * Nq_per_tile
            u1s.append(list(u1_full[i0:i1]))
            u2s.append(list(u2_full[j0:j1]))
            tile_ids.append(i*num_tiles_2+j)
    return u1s, u2s, tile_ids



###################################################################################################
# HELPERS
###################################################################################################

def check_if_files_complete(num_of_independent_runs, n, m_refs, defects, is_spin_ice):
    complete = True
    missing_files = []
    missing_run_ids = []
    missing_ns = []
    missing_m_refs = []
    missing_defects = []
    missing_partitions_to_ignore = []

    L = 6*n
    for m_ref in m_refs:
        for d in range(len(defects)):
            num_of_extra_particles, num_of_excitations = defects[d]
            for run_id in range(num_of_independent_runs):
                file_name = get_file_name(
                    run_id, m_ref, L, num_of_extra_particles, num_of_excitations, is_spin_ice)
                file_path = RAW_DATA_PATH / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
                    missing_run_ids.append(run_id)
                    missing_ns.append(n)
                    missing_m_refs.append(m_ref)
                    missing_defects.append(defects[d])
                    if is_spin_ice:
                        missing_partitions_to_ignore.append(0)
    if len(missing_files) != 0:
        complete = False
        print("Missing files:")
        for file in missing_files:
            print(file)
        print("\nrun_id:", missing_run_ids)
        print("n:", missing_ns)
        print("m_ref:", missing_m_refs)
        print("defects:", missing_defects)
        if is_spin_ice:
            print("partition_to_ignore:", missing_partitions_to_ignore)
        assert (
            len(missing_run_ids) == len(missing_ns) == len(missing_m_refs) == len(missing_defects)
            == len(missing_files))
        print(f"number of missing files: {len(missing_files)}\n")
    return complete



###################################################################################################
# DATA LOADER AND SAVER
###################################################################################################

def get_raw_data(file_name):
    file_path = RAW_DATA_PATH / file_name
    if not file_path.exists():
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)


def get_file_name(run_id, m_ref, L, num_of_extra_particles, num_of_excitations, is_spin_ice):
    if is_spin_ice:
        file_name = f"L{L}_HonSpinIce/{run_id}_MC_SAMPLING_{m_ref}m_ref_{L}L"
    else:
        file_name = f"L{L}/{run_id}_MC_SAMPLING_{m_ref}m_ref_{L}L"
    if num_of_extra_particles >= 0:
        file_name += f"_{num_of_extra_particles}num_of_extra_particles"
    else:
        file_name += f"_m{abs(num_of_extra_particles)}num_of_extra_particles"
    file_name += f"_{num_of_excitations}num_of_excitations_1000x1e05samples.pkl"
    return file_name


def save_structure_factor(Sq_grid, qx, qy, tile_id, inputs):
    data = {
        "inputs": inputs,
        "Sq_grid": Sq_grid,
        "qx": qx,
        "qy": qy,
        "tile_id": tile_id}

    L = inputs["L"]
    m_ref = inputs["m_ref"]
    num_of_extra_particles = inputs["num_of_extra_particles"]
    num_of_excitations = inputs["num_of_excitations"]
    is_spin_ice = inputs["is_spin_ice"]
    if tile_id is not None:
        data_path = PROCESSED_DATA_PATH_BIGSPACE
    else:
        data_path = PROCESSED_DATA_PATH
    
    file_name = get_structure_factor_file_name(
        tile_id, m_ref, L, num_of_extra_particles, num_of_excitations, is_spin_ice)
    file_path = data_path / f"{file_name}.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved as: {file_path}\n")


def get_structure_factor_file_name(
        tile_id, m_ref, L, num_of_extra_particles, num_of_excitations, is_spin_ice):
    file_name = "StructureFactor"
    if is_spin_ice:
        file_name += "_HonSpinIce"
    if tile_id is not None:
        file_name += f"_{tile_id}"
    if num_of_extra_particles >= 0:
        file_name += f"_{m_ref}m_ref_{L}L_{num_of_extra_particles}num_of_extra_particles"
    else:
        file_name += f"_{m_ref}m_ref_{L}L_m{abs(num_of_extra_particles)}num_of_extra_particles"
    file_name += f"_{num_of_excitations}num_of_excitations"
    return file_name



###################################################################################################
# EXECUTION
###################################################################################################

if __name__ == "__main__":
    check_completeness_only = True
    compute_correlations = True
    compute_structure_factors = False
    stitch_structure_factor_tiles_together = False

    # Correlations
    if compute_correlations:
        ns = [6, 10, 14, 18]
        num_of_runs = [100, 100, 10, 10]
        m_refs = [1, 2, 3]
        defects = [[0,0], [1,0], [2,0], [0,1]]
        for i, n in enumerate(ns):
            args = {
                "num_of_independent_runs": num_of_runs[i], 
                "n": n,
                "m_refs": m_refs,
                "defects": defects,
                "check_completeness_only": check_completeness_only}
            run(args)
                
            # For L=60 process Honeycomb Spin Ice data as well
            if n==10:
                args = {
                    "num_of_independent_runs": num_of_runs[i], 
                    "n": n,
                    "m_refs": m_refs,
                    "defects": [defects[0]],
                    "is_spin_ice": True,
                    "check_completeness_only": check_completeness_only}
                run(args)

    # Structure Factor
    u1_lim_full = [-1.3, 1.3]
    u2_lim_full = [-1.3, 1.3]
    num_tiles_1 = 3
    num_tiles_2 = 3
    num_of_tiles = num_tiles_1*num_tiles_2
    Nq_per_tile = 701
    m_refs = [1, 2, 3]
    L = 60
    is_spin_ices = [True, False]

    # Compute Structure Factor of Tiles (can be done in parallel)
    if compute_structure_factors:
        for m_ref in m_refs:
            for is_spin_ice in is_spin_ices:
                for tile_idx in range(num_tiles_1):
                    for tile_idy in range(num_tiles_2):
                        get_structure_factor(
                            m_ref, L, is_spin_ice, u1_lim_full, u2_lim_full, num_tiles_1, num_tiles_2, Nq_per_tile,
                            tile_idx, tile_idy, 
                            check_completeness_only=check_completeness_only)

    # Stitch Structure Factor Tiles
    if stitch_structure_factor_tiles_together:
        for m_ref in m_refs:
            for is_spin_ice in is_spin_ices:
                stitch_structure_factor_tiles(
                    num_tiles_1=num_tiles_1, num_tiles_2=num_tiles_2,
                    Nq_per_tile=Nq_per_tile, m_ref=m_ref, L=L, is_spin_ice=is_spin_ice, defect=[0, 0])
