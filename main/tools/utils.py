
# General Imports
import numpy as np

# Internal Imports
from main.tools.optional_njit import optional_njit



###################################################################################################
# TYPE VALIDATION
###################################################################################################

def validate_type(parameter, expected_type, name=None):
    """
    (Recursively) validate that ``parameter`` matches the structure and types described by 
    ``expected_type``.

    Parameters
    ----------
    parameter : any
        The object to check.
    expected_type : type, tuple of types, or list of types
        - If a single type or tuple of types: the parameter must be an instance of one of them.
        - If a list of types: each level of nesting corresponds to each list element.
          Example: [list, (int, float)] expects list[int] or list[float].
    name : str, optional
        Name of the variable.
    """
    typename = name or "parameter"

    # Non-nested (simple) case
    if not isinstance(expected_type, list):
        if not isinstance(parameter, expected_type):
            # Get readable names
            if isinstance(expected_type, tuple):
                expected_names = ", ".join(t.__name__ for t in expected_type)
            else:
                expected_names = expected_type.__name__

            raise ValueError(
                f"{typename} invalid, expected type `{expected_names}` "
                f"but got `{type(parameter).__name__}`.")
    else:
        # Outermost type should be list
        if not isinstance(parameter, expected_type[0]):
            raise ValueError(
                f"{typename} invalid, expected type `{expected_type[0].__name__}` "
                f"but got `{type(parameter).__name__}`.")

        # Last level reached
        if len(expected_type) == 1:
            return

        # Otherwise, recurse into each element
        for i, element in enumerate(parameter):
            validate_type(
                element, expected_type[1:], f"{typename}[{i}]")



###################################################################################################
# AVERAGE ARRAY OVER FIRST AXIS
###################################################################################################

def average_over_first_axis(array: np.ndarray):
    validate_type(array, np.ndarray, "array")
    N_axis0 = array.shape[0]
    mean = np.mean(array, axis=0)
    eps = np.sqrt(np.sum((array - mean)**2, axis=0) / (N_axis0 * (N_axis0 - 1)))
    return mean, eps



###################################################################################################
# NUMBA RELATED HELPERS
###################################################################################################

@optional_njit()
def ensure_capacity_2D(array:np.ndarray, used_rows:int, required_rows:int, new_row_size:int):
    storage_rows, num_cols = array.shape
    assert used_rows <= storage_rows
    if storage_rows >= required_rows:
        return array
    if new_row_size < required_rows:
        new_row_size = required_rows
    new_array = np.zeros((new_row_size, num_cols), dtype=array.dtype)
    new_array[:used_rows, :] = array[:used_rows, :] # copy entries
    return new_array

@optional_njit()
def ensure_capacity_1D(
    array:np.ndarray, used_entries:int, required_entries:int, new_entry_size:int):
    storage_entries = array.shape[0]
    assert used_entries <= storage_entries
    if storage_entries >= required_entries:
        return array
    if new_entry_size < required_entries:
        new_entry_size = required_entries
    new_array = np.zeros((new_entry_size,), dtype=array.dtype)
    new_array[:used_entries] = array[:used_entries] # copy entries
    return new_array



###################################################################################################
# NESTED LISTS TO NUMPY ARRAYS
###################################################################################################

def nested_list_to_padded_array(nested_list, fill_value=-1, dtype=np.int64) -> np.ndarray:
    """
    Convert a nested list into a fully padded numpy array.

    Parameters
    ----------
    nested_list : list
        Arbitrarily nested list of scalars (int, float, bool, etc.)
    fill_value : scalar
        Value used to pad missing elements
    dtype : numpy dtype
        Desired dtype of the resulting array
    """
    shape = get_max_shape(nested_list)
    arr = np.full(shape, fill_value, dtype=dtype)
    _fill_array_recursive(arr, nested_list)
    return arr

def _fill_array_recursive(arr, nested_list):
    """Recursively fill numpy array from a nested list."""
    if not isinstance(nested_list, list):
        return
    for i, sub_list in enumerate(nested_list):
        if isinstance(sub_list, list):
            _fill_array_recursive(arr[i], sub_list)
        else:
            arr[i] = sub_list

def get_max_shape(nested_list):
    """
    Determine the maximum shape of a nested list.

    Parameters
    ----------
    nested_list : list
        nested list
    """
    if not isinstance(nested_list, list) or len(nested_list) == 0:
        # Base case: not a list or empty
        return ()
    current_len = len(nested_list) # Length at this level
    # Compute shapes of all sub-lists (recursively)
    sub_shapes = [get_max_shape(sub) for sub in nested_list if isinstance(sub, list)]
    if not sub_shapes:
        # No deeper lists => stop recursion
        return (current_len,)
    # Find maximum shape among all sub-lists
    max_sub_shape = tuple(max(sizes) for sizes in zip(*_pad_shapes(sub_shapes)))
    return (current_len,) + max_sub_shape

def _pad_shapes(shapes):
    """pad shape tuples with 1s so zip(*shapes) works correctly."""
    max_len = max(len(s) for s in shapes)
    return [s + (1,) * (max_len - len(s)) for s in shapes]
