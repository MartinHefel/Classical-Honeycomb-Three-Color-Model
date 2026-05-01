
# General Imports
from pathlib import Path
import pickle
from tqdm import tqdm

# Internal Imports
from main import methods, model


###################################################################################################
# CONFIGURATION
###################################################################################################

# Path to the raw data folder
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")



###################################################################################################
# RUN
###################################################################################################

def run():
    get_monte_carlo_data_points()
    get_exact_enumeration_data_points()
    get_tensor_network_contraction_data_points()
    get_power_method_exponent()
    get_paulings_estimate()
    



###################################################################################################
# DATA FOR ZEOR POINT ENTROPY
###################################################################################################

def get_monte_carlo_data_points():
    file_names = [
        [
        "States_HoneycombRectangularUnitCell_1m_ref_3L1_2L2_all_field_24.pkl",
        "States_HoneycombRectangularUnitCell_1m_ref_6L1_3L2_all_field_288.pkl",
        "States_HoneycombRectangularUnitCell_1m_ref_9L1_4L2_all_field_6432.pkl",
        "States_HoneycombRectangularUnitCell_1m_ref_12L1_5L2_all_field_343008.pkl",
        "States_HoneycombRectangularUnitCell_1m_ref_3L1_3L2_all_field_60.pkl",
        "States_HoneycombRectangularUnitCell_1m_ref_6L1_4L2_all_field_912.pkl",
        "States_HoneycombUnitCell_1m_ref_6L1_6L2_all_field_168.pkl",
        "States_HoneycombUnitCell_1m_ref_9L1_9L2_all_field_60.pkl",
        "States_HoneycombUnitCell_1m_ref_12L1_12L2_all_field_2758128.pkl"],
        [
        "States_HoneycombRectangularUnitCell_2m_ref_3L1_2L2_all_field_219.pkl",
        "States_HoneycombRectangularUnitCell_2m_ref_6L1_3L2_all_field_176010.pkl",
        "States_HoneycombRectangularUnitCell_2m_ref_3L1_3L2_all_field_1398.pkl",
        "States_HoneycombRectangularUnitCell_2m_ref_6L1_4L2_all_field_4598799.pkl",
        "States_HoneycombUnitCell_2m_ref_3L1_3L2_all_field_78.pkl",
        "States_HoneycombUnitCell_2m_ref_6L1_6L2_all_field_153066.pkl"],
        [
        "States_HoneycombRectangularUnitCell_3m_ref_3L1_2L2_all_field_532.pkl",
        "States_HoneycombRectangularUnitCell_3m_ref_6L1_3L2_all_field_3464512.pkl",
        "States_HoneycombRectangularUnitCell_3m_ref_3L1_3L2_all_field_5596.pkl",
        "States_HoneycombUnitCell_3m_ref_3L1_3L2_all_field_146.pkl",
        "States_HoneycombUnitCell_3m_ref_6L1_6L2_all_field_2991448.pkl"]
    ]
    method = "monte_carlo"
    for i in range(len(file_names)):
        m_ref = None
        num_of_sites = []
        num_of_states = [] 
        L1s, L2s = [], []
        additional_info = {"unit_cell_name":[]}
        for file_name in file_names[i]:
            file_name = "States/"+file_name
            data = get_raw_data(file_name)
            if m_ref is None:
                m_ref = data["inputs"]["m_ref"]
            else:
                assert m_ref == data["inputs"]["m_ref"]
            L1, L2 = data["inputs"]["L1"], data["inputs"]["L2"]
            unit_cell_name = data["inputs"]["unit_cell_name"]
            if unit_cell_name == "HoneycombUnitCell":
                num_of_sites.append(2*L1*L2)
            else:
                num_of_sites.append(4*L1*L2)
            num_of_states.append(data["outputs"]["number_of_states"])
            L1s.append(L1)
            L2s.append(L2)
            additional_info["unit_cell_name"].append(unit_cell_name)
            
        save_data(
            method, m_ref, num_of_sites, num_of_states, L1s, L2s, 
            additional_info=additional_info)



def get_exact_enumeration_data_points():
    m_refs = [1, 2, 3]
    unit_cells = [
        model.unit_cell.HoneycombRectangularUnitCell(), model.unit_cell.HoneycombUnitCell()]
    
    unit_cell_id = [0, 1]
    Ls = [(3,2), (3,3)]
    
    method = "exact_enumeration"
    for m_ref in m_refs:
        num_of_sites = []
        num_of_states = [] 
        L1s, L2s = [], []
        additional_info = {"unit_cell_name":[]}
        for i in range(len(unit_cell_id)):
            unit_cell = unit_cells[unit_cell_id[i]]
            unit_cell_name = unit_cell.__class__.__name__
            L1, L2 = Ls[i]
            lattice = model.lattice.LatticeFromUnitCell(L1, L2, unit_cell)
            N_filling = lattice.get_N_filling(m_ref)
            states, _ = methods.hamiltonian.get_all_basis_states(
                m_ref, lattice=lattice, N_filling=N_filling, ground_states_only=True)
            num_of_sites.append(lattice.sites.N)
            num_of_states.append(states.number_of_states)
            L1s.append(L1)
            L2s.append(L2)
            additional_info["unit_cell_name"].append(unit_cell_name)
        save_data(
            method, m_ref, num_of_sites, num_of_states, L1s, L2s, 
            additional_info=additional_info)



def get_tensor_network_contraction_data_points():
    m_refs = [1, 2, 3]
    unit_cells = [
        model.unit_cell.HoneycombRectangularUnitCell(), model.unit_cell.HoneycombUnitCell()]
    
    unit_cell_id = [0, 0, 1]
    Ls = [(3,2), (3,3), (3,3)]
    
    method = "tensor_network_contraction"
    for m_ref in m_refs:
        num_of_sites = []
        num_of_states = [] 
        L1s, L2s = [], []
        additional_info = {"unit_cell_name":[]}
        for i in tqdm(range(len(unit_cell_id)), desc=f"m_ref={m_ref}"):
            unit_cell = unit_cells[unit_cell_id[i]]
            unit_cell_name = unit_cell.__class__.__name__
            L1, L2 = Ls[i]
            lattice = model.lattice.LatticeFromUnitCell(L1, L2, unit_cell)
            number_of_states, _ = methods.tensor_network.get_number_of_all_states_within_manifold(
                m_ref, lattice=lattice)
            num_of_sites.append(lattice.sites.N)
            num_of_states.append(number_of_states)
            L1s.append(L1)
            L2s.append(L2)
            additional_info["unit_cell_name"].append(unit_cell_name)
        save_data(
            method, m_ref, num_of_sites, num_of_states, L1s, L2s, 
            additional_info=additional_info)



def get_power_method_exponent():
    m_refs = [1, 2, 3]
    ls = [2, 3, 4, 5]
    unit_cell = model.unit_cell.HoneycombUnitCell()
    chi = 30
    method = "power_method"
    for m_ref in m_refs:
        alphas = []
        for l in tqdm(ls, desc=f"m_ref={m_ref}"):
            alpha, diagnostics = methods.tensor_network.get_alpha_from_transfer_matrix(
                L=l, chi=chi, lattice=unit_cell, A_transfer_kwargs={"m_ref":m_ref})
            assert diagnostics["converged"]
            alphas.append(alpha)
    
        data = {
            "method": method,
            "m_ref": m_ref,
            "ls": ls,
            "alphas": alphas,
            "diagnostics": diagnostics}
        file_name = f"ZeroPointEntropy_{method}_{m_ref}m"
        file_path = PROCESSED_DATA_PATH / f"{file_name}.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved as: {file_path}\n")



def get_paulings_estimate():
    def get_variant(variant_number:int):
        pl_id_to_site_ids_donut = [
            [0,1,2,3,4,5], [4,3,6,7,8,9], [13,9,8,10,11,12],
            [16,17,13,12,14,15], [19,20,21,17,16,18], [22,23,0,5,21,20]]
        pl_id_to_site_ids_L = [
            [0,1,2,3,4,5], [4,3,6,7,8,9], [8,7,10,11,12,13],
            [14,13,12,15,16,17],[18,19,14,17,20,21], [22,23,18,21,24,25]]
        
        if variant_number == 0:
            pl_id_to_site_ids_tile = [[0,1,2,3,4,5]]
            pl_id_to_site_ids_gap = pl_id_to_site_ids_donut
            edge_mappings = [
                [[21,5,4,9,13,17], [0,1,2,3,4,5]],
                [[23,0,1], [5,4,3]],
                [[2,3,6], [0,5,4]],
                [[7,8,10], [1,0,5]],
                [[11,12,14], [2,1,0]],
                [[15,16,18], [3,2,1]],
                [[19,20,22], [4,3,2]]]
            num_of_cell_plaquettes = 3
        elif variant_number == 1:
            pl_id_to_site_ids_tile = [
                [0,1,2,3,4,5], [4,3,6,7,8,9],
                [12,5,4,9,10,11], [14,15,0,5,12,13]]
            pl_id_to_site_ids_gap = pl_id_to_site_ids_L
            pl_id_to_site_ids_gap[-1] = [22,23,0,5,24,25]
            edge_mappings = [
                [[25,24,4,9,8,13,14,19,18,23], [15,0,1,2,3,6,7,8,9,10]],
                [[21,20,17,16], [15,0,1,2]],
                [[15,12,11], [13,14,15]],
                [[10,7,6,3,2],[10,11,12,13,14]],
                [[1,0,23,22],[7,8,9,10]]]
            num_of_cell_plaquettes = 8
        elif variant_number == 2:
            pl_id_to_site_ids_tile = pl_id_to_site_ids_donut
            pl_id_to_site_ids_tile.append([21,5,4,9,13,17])
            pl_id_to_site_ids_tile.append([8,7,24,25,26,10])
            pl_id_to_site_ids_tile.append([28,29,22,20,19,27])
            pl_id_to_site_ids_gap = pl_id_to_site_ids_L
            edge_mappings = [
                [[0,5,4,9,8,13,14,19,18,23,22], [2,3,6,7,24,25,26,10,11,12,14]],
                [[25,24,21,20,17,16], [30,22,23,0,1,2]],
                [[15,12,11], [28,29,30]],
                [[10,7,6,3,2,1], [14,15,16,18,19,28]]]
            num_of_cell_plaquettes = 15
        return pl_id_to_site_ids_gap, pl_id_to_site_ids_tile, edge_mappings, num_of_cell_plaquettes

    m_refs = [1, 2, 3]
    method = "pauling_like_estimate"
    for m_ref in m_refs:
        num_of_tile_states = []
        num_of_gap_states = []
        gap_tile_ratios = []
        cell_sizes = []
        alpha_maxs = []
        Ps = []
        for i in range(3):
            pls_gap, pls_tile, edge_mappings, num_of_cell_plaquettes = get_variant(i)
            
            alpha_max, P, diagnostics = methods.paulings_estimate.get_pauling_like_estimate(
            m_ref, edge_mappings, num_of_cell_plaquettes, sites_per_pl=2, refer_to_sites=True,
            ex_pl_id_to_site_ids_tile=pls_tile, ex_pl_id_to_site_ids_gap=pls_gap)
            
            cell_sizes.append(num_of_cell_plaquettes)
            alpha_maxs.append(alpha_max)
            Ps.append(P)
            num_of_tile_states.append(diagnostics["number_of_tile_states"])
            num_of_gap_states.append(diagnostics["number_of_gap_states"])
            gap_tile_ratios.append( 1-len(pls_tile)/num_of_cell_plaquettes)
    
        data = {
            "method": method,
            "m_ref": m_ref,
            "cell_sizes": cell_sizes,
            "alpha_maxs": alpha_maxs,
            "Ps": Ps,
            "num_of_tile_states": num_of_tile_states,
            "num_of_gap_states": num_of_gap_states,
            "gap_tile_ratios": gap_tile_ratios}
        file_name = f"ZeroPointEntropy_{method}_{m_ref}m"
        file_path = PROCESSED_DATA_PATH / f"{file_name}.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved as: {file_path}\n")



###################################################################################################
# DATA SAVER
###################################################################################################

def save_data(
        method:str, m_ref:int, num_of_sites:list[int], num_of_states:list[int], 
        L1s:list[int], L2s:list[int], additional_info:dict=None
):
    assert len(num_of_sites) == len(num_of_states) == len(L1s) == len(L2s)
    data = {
        "method": method,
        "m_ref": m_ref,
        "num_of_sites": num_of_sites,
        "num_of_states": num_of_states,
        "additional_info": additional_info,
        "L1s": L1s,
        "L2s": L2s}

    # Save data
    file_name = f"ZeroPointEntropy_{method}_{m_ref}m"
    file_path = PROCESSED_DATA_PATH / f"{file_name}.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved as: {file_path}\n")


###################################################################################################
# DATA LOADER
###################################################################################################

def get_raw_data(file_name):
    file_path = RAW_DATA_PATH / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)



###################################################################################################
# EXECUTION
###################################################################################################

if __name__ == "__main__":
    run()