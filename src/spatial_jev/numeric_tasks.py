"""Input-selected numeric codebooks; never interpret counts or degrees as meters."""

import math

COUNT_CODEBOOK = {"version": "integer-32x32-v1", "unit": "count", "fine_bins": 32,
                  "maximum": 1023}
ANGLE_CODEBOOK = {"version": "degrees-30x32-v1", "unit": "deg", "fine_bins": 32,
                  "edges_m": list(range(0, 361, 30)), "overflow_representative_m": 360.0}
# edges_m is the historical controller's internal key, not the physical unit.
# Public answer spaces, prompts, validators and decoded units remain explicit.


def codebook_for(inp, metric):
    unit = inp["answer_space"].get("unit", "m")
    if unit == "m":
        return metric
    if unit == "count":
        return COUNT_CODEBOOK
    if unit == "deg":
        return ANGLE_CODEBOOK
    raise ValueError(f"Unsupported numeric unit: {unit}")


def numeric_path(value, codebook):
    from .hierarchy import scalar_path
    if codebook.get("unit", "m") != "count":
        if codebook.get("unit") == "deg" and not 0 <= value <= 360:
            raise ValueError("Angle outside [0, 360]")
        return scalar_path(value, codebook)
    if isinstance(value, bool) or not math.isfinite(value) or int(value) != value or not 0 <= value <= 1023:
        raise ValueError("Count outside integer codebook [0, 1023]")
    return [int(value) // 32, int(value) % 32]


def numeric_done(path, codebook):
    return len(path) == 2 or (codebook.get("unit", "m") != "count" and path[0] in (0, len(codebook["edges_m"])))


def decode_numeric(path, codebook):
    from .hierarchy import decode_scalar
    if codebook.get("unit", "m") != "count":
        return decode_scalar(path, codebook)
    if len(path) != 2 or any(not 0 <= n < 32 for n in path):
        raise ValueError("Invalid integer decision path")
    return 32 * path[0] + path[1]


def numeric_prompt(inp, root, codebook):
    """Non-metric prompts. Metric prompts intentionally keep their old exact bytes."""
    if codebook["unit"] == "count":
        options = ([f"count {32*i} through {32*i+31} (inclusive)" for i in range(32)]
                   if root is None else [f"count exactly {32*root+i}" for i in range(32)])
        instruction = "Answer the counting question. Select the integer count or inclusive count range."
    else:
        from .hierarchy import scalar_interval
        edges = codebook["edges_m"]
        if root is None:
            options = ["exactly zero degrees"]
            options += [f"{lo:g} to {hi:g} degrees" for lo, hi in zip(edges, edges[1:])]
            options[1] = "greater than zero and less than 30 degrees"
            options += ["greater than 360 degrees (out of supported range)"]
        else:
            lo, hi = scalar_interval(root, codebook)
            step = (hi - lo) / codebook["fine_bins"]
            options = [f"{lo+i*step:.6g} to {lo+(i+1)*step:.6g} degrees" for i in range(32)]
        instruction = ("Estimate the angle in degrees, preserving the reference frame and rotation "
                       "direction specified in the question. Ranges include the lower endpoint and "
                       "exclude the upper, except that 360 is included. Exact zero has its own class.")
    return (f"Question: {inp['question']}\n{instruction}\nOptions:\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(options))
            + "\nSelect exactly one option."), len(options)
