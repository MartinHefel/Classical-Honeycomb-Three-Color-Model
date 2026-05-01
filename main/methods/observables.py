
# General Imports
import numpy as np

# Internal Imports
from main.tools.utils import validate_type
from main.tools.optional_njit import optional_njit
from main.model.states import StateCollection
from main.model.lattice import Lattice



###################################################################################################
# TWO-POINT CORRELATION FUNCTION
###################################################################################################

def measure_two_point_correlation_function(
        states:StateCollection, connected:bool=True, chunk_size=int(1e5)) -> np.ndarray:
    r""" Compute the (connected) two-point correlation function
    
    .. math::
        C_{ij} = ⟨n_i n_j⟩ - ⟨n_i⟩ ⟨n_j⟩,

    where ⟨.⟩ denotes the expectation value taken over the provided states.

    Parameters
    ----------
    states : StateCollection
        Collection of states from which the correlation function is evaluated.
    connected : bool
        If ``True``, compute the connected correlation function (i.e., subtract the product of 
        one-point expectation values). If ``False``, return the unconnected correlator ⟨n_i n_j⟩.
    chunk_size : int
        Block size used for intermediate matrix computations. Smaller values reduce memory usage.
        This parameter does not affect the final result.

    Returns
    -------
    C_ij : np.ndarray
        Matrix containing the two-point correlation function between sites ``i`` and ``j``.
    """
    validate_type(chunk_size, int, "chunk_size")
    
    if isinstance(states, StateCollection):
        states_array = states.as_array()
    else:
        validate_type(states, np.ndarray, "states")
        states_array = states
    num_of_states, num_of_sites = states_array.shape # not necessarily sites, can also be plaquettes

    chunk_size = min(chunk_size, num_of_states)

    # Compute mean occupations <n_i>
    if connected:
        n_mean = np.zeros((num_of_sites,1), dtype=np.float64)
        for chunk_start in range(0, num_of_states, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_of_states)
            chunk = np.asarray(states_array[chunk_start:chunk_end,:], dtype=np.float64)
            n_mean[:,0] += np.sum(chunk, axis=0)
        n_mean /= num_of_states
    else:
        n_mean = None

    # Compute <n_i n_j>
    ni_nj_mean = np.zeros((num_of_sites, num_of_sites), dtype=np.float64)
    for chunk_start in range(0, num_of_states, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_of_states)
        chunk = np.asarray(states_array[chunk_start:chunk_end,:], dtype=np.float64)
        ni_nj_mean += chunk.T @ chunk
    ni_nj_mean /= num_of_states

    # C_ij (num_of_ids x num_of_ids) 
    if connected:
        C_ij = ni_nj_mean - n_mean @ n_mean.T
    else:
        C_ij = ni_nj_mean
    return C_ij



def C_ij_to_C_kr(
        C_ij:np.ndarray, patterns:list[list[int]], k_to_i:list[int|None]
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Map correlations ``C_ij`` to ``C_kr`` using patterns.

    For each pattern index ``k``, a source index ``i = k_to_i[k]`` is selected. If ``i`` is not
    ``None``, the correlations are transformed as ``C_kr[row, :] = C_ij[i, patterns[k]]``,
    where ``row`` enumerates only the valid (non-``None``) entries of ``k_to_i``. Patterns with
    ``k_to_i[k] is None`` are skipped.

    This remaps the original ``j``-index to a new ``r``-index via a pattern and filters out 
    unwanted mappings.

    Parameters
    ----------
    C_ij : np.ndarray
        Correlation array of shape ``(num_of_ids, num_of_corrs)``.
        Typically ``num_of_ids == num_of_corrs`` unless the data have been truncated or
        transformed beforehand.

    patterns : list[list[int]]
        List of pattern lists defining the mapping ``j -> r``.
        All patterns must have the same length ``len(patterns[k])``.

    k_to_i : list[int | None]
        Mapping from pattern index ``k`` to source index ``i`` in ``C_ij``.
        If ``k_to_i[k]`` is ``None``, the corresponding patterns is ignored and does not
        contribute to the output.

    Returns
    -------
    rs : np.ndarray
        Array ``[0, 1, ..., R-1]`` representing the new ``r``-indices along each pattern.
        ``R`` is the length of each pattern (all patterns must have the same length).
    C_kr : np.ndarray
        Transformed and filtered correlation array of shape ``(num_of_valid_ids, num_of_rs)``, 
        where ``num_of_valid_ids`` is the number of entries in ``k_to_i`` that are not ``None``.
    """

    validate_type(C_ij, np.ndarray, "C_ij")
    validate_type(patterns, [list, list, int], "patterns")
    validate_type(k_to_i, list, "k_to_i")
    assert C_ij.ndim == 2
    num_of_ids, num_of_corrs = C_ij.shape

    num_valid_ids = sum(1 for i in k_to_i if i is not None)
    pattern_length = None
    for ids in patterns:
        if pattern_length is None:
            pattern_length = len(ids)
        else:
            assert pattern_length == len(ids)
    C_kr = np.empty((num_valid_ids, pattern_length), dtype=C_ij.dtype)

    row_id = 0
    for k, i in enumerate(k_to_i):
        if i is not None:
            C_kr[row_id, :] = C_ij[i, patterns[k]]
            row_id += 1
    rs = np.arange(0,pattern_length)
    return rs, C_kr



###################################################################################################
# STRUCTURE FACTOR
###################################################################################################

def compute_static_structure_factor_grid(
        C_ij:np.ndarray, lattice:Lattice, Nq1:int=101, Nq2:int=101, 
        u1_lim:list[float,float]=[-0.5, 0.5], u2_lim:list[float,float]=[-0.5, 0.5], 
        us_1:list[float]=None, us_2:list[float]=None, 
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute the static structure factor ``S(q)`` on a regular 2D grid in reciprocal space.

    This function constructs a uniform grid of wave vectors using the lattice's reciprocal vectors,
    evaluates the static structure factor at all points, and reshapes it into a 2D array.

    Parameters
    ----------
    C_ij : np.ndarray
        Two-point connected correlation matrix of shape ``(num_of_ids, num_of_ids)``.
    lattice : Lattice
        The lattice object providing coordinates and unit cell vectors.
    Nq1 : int, optional
        Number of grid points along the first reciprocal lattice direction.
        Ignored if ``us_1`` is provided. Default is ``101``.
    Nq2 : int, optional
        Number of grid points along the second reciprocal lattice direction.
        Ignored if ``us_2`` is provided. Default is ``101``.
    u1_lim : list of two floats, optional
        Minimum and maximum fractional coordinates along the first reciprocal lattice vector.
        Default is ``[-0.5, 0.5]``.
    u2_lim : list of two floats, optional
        Minimum and maximum fractional coordinates along the second reciprocal lattice vector.
        Default is ``[-0.5, 0.5]``.
    us_1 : list or np.ndarray, optional
        Explicit fractional coordinates along the first reciprocal lattice vector. 
        If provided, overrides ``u1_lim`` and ``Nq1``.
    us_2 : list or np.ndarray, optional
        Explicit fractional coordinates along the second reciprocal lattice vector. 
        If provided, overrides ``u2_lim`` and ``Nq2``.

    Returns
    -------
    Sq_grid : np.ndarray
        2D array of the static structure factor evaluated on the grid, shape ``(Nq1, Nq2)``.
        Each element corresponds to the value at ``(qx[i,j], qy[i,j])``.
    qx : np.ndarray
        2D array of x-components of the wave vectors, shape ``(Nq1, Nq2)``.
    qy : np.ndarray
        2D array of y-components of the wave vectors, shape ``(Nq1, Nq2)``.
    """
    validate_type(lattice, Lattice, "lattice")
    validate_type(Nq1, int, "Nq1")
    validate_type(Nq2, int, "Nq2")
    validate_type(u1_lim, [list, float], "u1_lim")
    validate_type(u2_lim, [list, float], "u2_lim")
    if (us_1 is None) != (us_2 is None):
        raise ValueError("us_1 and us_2 must either both be provided or both be None")
    elif us_1 is not None and us_2 is not None:
        validate_type(us_1, [list, float], "us_1")
        validate_type(us_2, [list, float], "us_2")
        Nq1 = len(us_1)
        Nq2 = len(us_2)
    else:
        assert len(u1_lim) == 2 and len(u2_lim) == 2
    # remaining parameters checked in compute_static_structure_factor

    a1, a2 = lattice.sites.a1, lattice.sites.a2
    area = a1[0]*a2[1]-a1[1]*a2[0]
    b1 = 2*np.pi*np.array([ a2[1], -a2[0]])/area
    b2 = 2*np.pi*np.array([-a1[1],  a1[0]])/area
    if us_1 is None and us_2 is None:
        us_1 = np.linspace(u1_lim[0], u1_lim[1], Nq1)
        us_2 = np.linspace(u2_lim[0], u2_lim[1], Nq2)
    else:
        us_1 = np.array(us_1)
        us_2 = np.array(us_2)
    Us_1, Us_2 = np.meshgrid(us_1, us_2, indexing="ij")
    qx = Us_1 * b1[0] + Us_2 * b2[0]
    qy = Us_1 * b1[1] + Us_2 * b2[1]
    q_points = (np.stack([qx.ravel(), qy.ravel()], axis=1)).T

    S_q = compute_static_structure_factor(C_ij, q_points, lattice)
    
    Sq_grid = S_q.reshape(Nq1, Nq2)
    return Sq_grid, qx, qy



def compute_static_structure_factor(
        C_ij:np.ndarray, qs:np.ndarray, lattice:Lattice, batch_size:int=256
) -> np.ndarray:
    r"""
    Compute the static structure factor ``S(q)`` from a given two-point connected correlation 
    matrix ``C_ij``. The static structure factor is defined as

    .. math::

        S(q) = 1/N \sum_{i,j} C_{ij} \cos(q \cdot (x_j-x_i) )

    where ``C_ij`` is the connected correlation matrix and ``x_i`` are the Cartesian coordinates
    of the lattice nodes, and ``N`` is the number of nodes.

    The coordinates used are determined automatically from the size of 
    ``C_ij``:

    - If ``C_ij.shape[0]==lattice.sites.N``, site coordinates 
    (``lattice.sites.coords``) are used.
    - If ``C_ij.shape[0]==lattice.plaquettes.N``, plaquette-center coordinates 
    (``lattice.plaquettes.center_coords``) are used.

    Parameters
    ----------
    C_ij : np.ndarray
        Two-point connected correlation matrix of shape ``(num_of_ids, num_of_ids)``.
    qs : np.ndarray
        Array of wave vectors with shape ``(2, num_of_qs)``, where each column corresponds to a 
        Cartesian momentum vector ``q = (q_x, q_y)``.
    lattice : Lattice
        The lattice object providing coordinates.
    batch_size : int, optional
        Number of q-points to process at a time. Defaults to ``256``.
        Smaller batch sizes use less RAM but may slightly increase computation time.

    Returns
    -------
    S_q : np.ndarray
        Static structure factor evaluated at each momentum ``q``, with shape ``(num_of_qs,)``.
    """
    validate_type(C_ij, np.ndarray, "C_ij")
    validate_type(qs, np.ndarray, "qs")
    validate_type(lattice, Lattice, "lattice")

    assert C_ij.ndim == 2
    assert C_ij.shape[0] == C_ij.shape[1]
    assert qs.ndim == 2
    assert qs.shape[0] == 2
    # \/ if this is the case, add a variable to choose coordinates
    assert lattice.sites.N != lattice.plaquettes.N 

    num_of_ids = C_ij.shape[0]
    if num_of_ids == lattice.sites.N:
        coords = lattice.sites.coords # (num_of_ids, 2)
    elif num_of_ids == lattice.plaquettes.N:
        coords = lattice.plaquettes.center_coords # (num_of_ids, 2)
    else:
        raise ValueError(
            f"Size mismatch: C_ij has shape {C_ij.shape}, "
            f"but lattice has {lattice.sites.N} sites and {lattice.plaquettes.N} plaquettes.")
    
    num_of_qs = qs.shape[1]
    S_q = np.empty(num_of_qs)

    for start in range(0, num_of_qs, batch_size):
        end = min(start + batch_size, num_of_qs)
        qs_batch = qs[:, start:end]
        phase = coords @ qs_batch
        phi = np.exp(1j * phase)
        C_phi = C_ij @ phi
        S_q[start:end] = np.real(np.sum(phi.conj()*C_phi, axis=0))
    return S_q / num_of_ids
