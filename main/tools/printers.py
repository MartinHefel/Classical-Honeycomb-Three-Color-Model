
# General Imports
from typing import Any
import numpy as np

# Internal Imports
from main.tools.optional_njit import optional_njit



###################################################################################################
# PRINT TO CONSOLE
###################################################################################################

@optional_njit()
def print_progress_to_console(i:int, L:int, num_updates:int=10):
    if num_updates == -1:
        print(f"  Progress: {i+1}/{L}")
    elif L <= num_updates:
        print(f"  Progress: {i+1}/{L}")
    else:
        step = L // num_updates
        if (i + 1) % step == 0 or i == L - 1 or i==0:
            print(f"  Progress: {i+1}/{L}")



###################################################################################################
# MAKE TABLE FROM DATA
###################################################################################################

def make_table(headers: list[str], rows: list[list[str]], max_width: int = 100) -> str:

    # Determine column widths based on header and content
    n_cols = len(headers)
    col_widths = [
        max(len(h), max((len(str(row[i])) for row in rows), default=0))
        for i, h in enumerate(headers)]

    total_width = sum(col_widths) + 3 * (n_cols - 1)
    # Scale down if too wide
    if total_width > max_width:
        scale = max_width / total_width
        col_widths = [max(8, int(w * scale)) for w in col_widths]

    # Format header
    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep = "-" * len(header_line)
    lines = ["\n"+header_line, sep]

    # Format rows with wrapping
    for row in rows:
        # Convert all to string
        cells = [str(c) for c in row]
        # Wrap each cell individually
        wrapped_cells = [
            [cells[i][j:j + col_widths[i]] for j in range(0, len(cells[i]), col_widths[i])]
            or [""]
            for i in range(n_cols)]
        n_lines = max(len(c) for c in wrapped_cells)
        for l in range(n_lines):
            line = " | ".join(
                f"{(wrapped_cells[i][l] if l < len(wrapped_cells[i]) else ''):<{col_widths[i]}}"
                for i in range(n_cols))
            lines.append(line)
    lines.append("")
    return "\n".join(lines)


def make_table_data(data: list[list[Any]], float_fmt: str = ".3f") -> list[list[str]]:
    str_rows = []
    for row in data:
        str_row = []
        for val in row:
            # Handle numeric types
            if isinstance(val, float):
                str_row.append(f"{val:{float_fmt}}")
            elif isinstance(val, (int, np.integer)):
                str_row.append(f"{val:d}")

            # Handle None
            elif val is None:
                str_row.append("-")

            # Handle ListWithShifts
            elif hasattr(val, "entries") and hasattr(val, "x_shifts") and hasattr(val, "y_shifts"):
                parts = []
                for nid, sx, sy in zip(val.entries, val.x_shifts, val.y_shifts):
                    if sx != 0 or sy != 0:
                        parts.append(f"{nid}(sx={sx},sy={sy})")
                    else:
                        parts.append(str(nid))
                str_row.append(", ".join(parts))

            # Handle lists/tuples
            elif isinstance(val, (list, tuple)):
                str_row.append(", ".join(map(str, val)))

            # Fallback
            else:
                str_row.append(str(val))

        str_rows.append(str_row)
    return str_rows
