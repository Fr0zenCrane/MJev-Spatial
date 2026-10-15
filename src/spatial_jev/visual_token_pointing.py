"""Direct question-Q / image-token-K pointing on Qwen's merged visual grid.

The candidates are LLM image-token positions, not raw ViT keys (a different
projection space). Geometry follows image_grid_thw after spatial merging, in
row-major order, with no crop, label-dependent input, or learned coordinate head.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F

from .point_probability_map import component_points
from .roi_replay import REPLAY_VARIANTS

VISUAL_TOKENS = "visual_tokens"
ORIGINAL_IMAGE = "original_image_nine_way"


def validate_pointing_config(config):
    variant = config.get("point_variant", "crop_refill")
    if variant not in ("crop_refill", VISUAL_TOKENS, ORIGINAL_IMAGE, *REPLAY_VARIANTS):
        raise ValueError(f"Unsupported KEV pointing method: {variant}")
    if variant == ORIGINAL_IMAGE and config.get("point_depth", 3) != 3:
        raise ValueError("Original-image nine-way pointing requires three decisions")
    if variant in REPLAY_VARIANTS and (
            config.get('point_depth', 3) != 3 or config.get('readout_kind') != 'native_last_qk_mean'
                or config.get('readout_layer_indices') is not None or config.get('readout_layer_index', 31) != 31
            or config.get('shared_prefix', False) or config.get('gradient_checkpointing', False)):
        raise ValueError('ROI replay fixes L32 mean, three decisions and no persistent cache/checkpointing')
    if variant != VISUAL_TOKENS:
        return
    if config.get("point_depth", 1) != 1:
        raise ValueError("Visual-token pointing is a single decision, not a crop tree")
    if config.get("shared_prefix", False):
        raise ValueError("Visual-token keys require the full causal row; shared_prefix is unsupported")
    if config.get("point_loss", "uniform_cell_ce") != "uniform_cell_ce":
        raise ValueError("Visual-token pointing requires uniform_cell_ce")
    if config.get("point_decoder", "argmax") not in ("argmax", "components"):
        raise ValueError("Unknown visual-token point decoder")
    if not 0 < config.get("point_component_threshold", 0.1) <= 1:
        raise ValueError("Invalid visual-token component threshold")


def image_token_candidates(ids, image_token_id, grid_thw, merge_size):
    """One original image only; reject video, missing tokens and ambiguous grids."""
    t, h, w = [int(v) for v in grid_thw]
    if merge_size < 1 or t != 1 or min(h, w) < 1 or h % merge_size or w % merge_size:
        raise ValueError("Expected one spatially mergeable still-image grid")
    shape = (h // merge_size, w // merge_size)
    positions = [i for i, token in enumerate(ids) if token == image_token_id]
    if len(positions) != math.prod(shape) or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("Image-token positions do not match the merged visual grid")
    return positions, shape


def target_cells(points, shape):
    """Map normalized training labels to occupied cells, without modifying inputs."""
    if not isinstance(points, (list, tuple)) or not points:
        raise ValueError("Point supervision must contain at least one normalized point")
    height, width = shape
    if min(height, width) < 1:
        raise ValueError("Empty visual grid")
    cells = set()
    for point in points:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or not 0 <= v <= 1 for v in point)):
            raise ValueError("Point supervision must use finite normalized coordinates")
        x, y = point
        col, row = min(int(x * width), width - 1), min(int(y * height), height - 1)
        cells.add(row * width + col)
    return sorted(cells)


def decision_losses(logits, packed):
    """Ordinary CE unchanged; multiple occupied visual cells share one QA's mass.

    Sparse targets avoid 0 * -inf on padded candidate slots. Coordinates in the
    same visual cell are intentionally indistinguishable at this resolution.
    """
    targets = packed["targets"].to(logits.device)
    if "visual_target_owner" not in packed:
        return F.cross_entropy(logits, targets, reduction="none")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    losses = -log_probs.gather(1, targets[:, None]).squeeze(1)
    owners = packed["visual_target_owner"].to(logits.device)
    slots = packed["visual_target_slot"].to(logits.device)
    mass = packed["visual_target_mass"].to(logits.device)
    visual_losses = torch.zeros_like(losses).index_add(0, owners, -log_probs[owners, slots] * mass)
    selected = owners.unique()
    return losses.index_copy(0, selected, visual_losses[selected])


def decode_visual_points(logits, shape, decoder="argmax", relative_threshold=0.1):
    """Decode only model scores and known geometry; never receive GT/count/masks."""
    values = np.asarray(logits, dtype=np.float64)
    height, width = shape
    if min(height, width) < 1 or values.shape != (height * width,) or not np.isfinite(values).all():
        raise ValueError("Expected one finite score per merged visual token")
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    grid = probabilities.reshape(height, width)
    if decoder == "argmax":
        index = int(values.argmax())
        y, x = divmod(index, width)
        points = [[(x + 0.5) / width, (y + 0.5) / height]]
    elif decoder == "components":
        components = component_points(grid, relative_threshold=relative_threshold, connectivity=8)["components"]
        components.sort(key=lambda c: (-c["peak_probability"], *c["peak_cell"]))
        points = [[(c["peak_cell"][1] + 0.5) / width, (c["peak_cell"][0] + 0.5) / height]
                  for c in components]
    else:
        raise ValueError(f"Unknown visual-token point decoder: {decoder}")
    return {"prediction": points[0], "points": points, "grid_hw": list(shape),
            "probabilities": grid.tolist(), "pointing_method": VISUAL_TOKENS,
            "point_decoder": decoder, "point_component_threshold": relative_threshold if decoder == "components" else None}
