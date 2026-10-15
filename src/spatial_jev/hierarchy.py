"""Deterministic scalar/point classification codebooks and visual ROI geometry."""

import bisect
import math

import numpy as np

# User-specified order: 正上、左上、右上、正中、左中、右中、正下、左下、右下.
POINT_NAMES = ("top-center", "top-left", "top-right", "center", "middle-left",
               "middle-right", "bottom-center", "bottom-left", "bottom-right")
POINT_RC = ((0, 1), (0, 0), (0, 2), (1, 1), (1, 0), (1, 2), (2, 1), (2, 0), (2, 2))
ROOT_BOX = (0.0, 0.0, 1.0, 1.0)


def child_box(box, choice):
    if not 0 <= choice < 9:
        raise ValueError("Invalid nine-way choice")
    x0, y0, x1, y1 = box
    row, col = POINT_RC[choice]
    w, h = (x1 - x0) / 3, (y1 - y0) / 3
    return (x0 + col * w, y0 + row * h, x0 + (col + 1) * w, y0 + (row + 1) * h)


def point_label(xy, box):
    x0, y0, x1, y1 = box
    x, y = xy
    if not all(math.isfinite(v) for v in xy) or not (x0-1e-9 <= x <= x1+1e-9
                                                       and y0-1e-9 <= y <= y1+1e-9):
        raise ValueError("Point outside parent")
    col = min(2, max(0, math.floor((x - x0) / (x1 - x0) * 3 + 1e-12)))
    row = min(2, max(0, math.floor((y - y0) / (y1 - y0) * 3 + 1e-12)))
    return POINT_RC.index((row, col))


def point_path(xy, depth=3):
    box, path = ROOT_BOX, []
    for _ in range(depth):
        choice = point_label(xy, box)
        path.append(choice)
        box = child_box(box, choice)
    return path


def decode_point(path):
    box = ROOT_BOX
    for choice in path:
        box = child_box(box, choice)
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def scalar_fine_bins(root_choice, codebook):
    """Refinement cardinality is fixed by the selected root, never by the target."""
    if not 1 <= root_choice < len(codebook["edges_m"]):
        raise ValueError("Zero/overflow has no refinement candidates")
    by_root = codebook.get("fine_bins_by_root")
    if by_root is not None:
        if len(by_root) != len(codebook["edges_m"]) - 1:
            raise ValueError("Refinement counts must cover every positive root")
        fine = by_root[root_choice - 1]
    else:
        fine = codebook["fine_bins"]
    if type(fine) is not int or not 1 <= fine <= 32:
        raise ValueError("Refinement counts must be integers in [1, 32]")
    return fine


def scalar_path(value, codebook):
    edges = codebook["edges_m"]
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid nonnegative scalar")
    if value == 0:
        return [0]
    if value > edges[-1]:
        return [len(edges)]  # Explicit overflow, never silently clipped.
    bucket = min(bisect.bisect_right(edges, value) - 1, len(edges) - 2)
    fine = scalar_fine_bins(bucket + 1, codebook)
    lo, hi = edges[bucket:bucket+2]
    local = min(fine - 1, max(0, int((value - lo) / (hi - lo) * fine)))
    return [bucket + 1, local]


def scalar_interval(root_choice, codebook):
    edges = codebook["edges_m"]
    if not 1 <= root_choice < len(edges):
        raise ValueError("Zero/overflow has no refinement interval")
    return edges[root_choice - 1], edges[root_choice]


def decode_scalar(path, codebook):
    if path[0] == 0:
        return 0.0
    if path[0] == len(codebook["edges_m"]):
        return codebook["overflow_representative_m"]
    fine = scalar_fine_bins(path[0], codebook)
    if len(path) != 2 or not 0 <= path[1] < fine:
        raise ValueError("Missing/invalid scalar refinement")
    lo, hi = scalar_interval(path[0], codebook)
    return lo + (path[1] + .5) * (hi - lo) / fine


def decode_scalar_expectation(root_choice, fine_logits, codebook):
    """Conditional mean of fine-bin centers, T=1, within one argmax coarse root.

    No averaging across coarse roots and no target-dependent decisions. Reserved
    zero/overflow roots retain their existing exact representatives.
    """
    if root_choice in (0, len(codebook["edges_m"])):
        if len(fine_logits):
            raise ValueError("Zero/overflow must not have a second-stage distribution")
        return decode_scalar([root_choice], codebook)
    fine = scalar_fine_bins(root_choice, codebook)
    logits = np.asarray(fine_logits, dtype=np.float64)
    if logits.shape != (fine,) or not np.isfinite(logits).all():
        raise ValueError("Expectation requires exactly the finite valid fine logits")
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    centers = np.asarray([decode_scalar([root_choice, i], codebook) for i in range(fine)])
    return float(probabilities @ centers)


def visual_token_boxes(metadata, image_grids, image_num_crops, patches_per_crop):
    """Map pooled tokens to normalized original-image support boxes, including padding.

    Uses native subpatch_mapping (stretched stitched crop grid), not token-index slicing.
    Each image's global crop and high-resolution crop mapping have their own geometry.
    """
    pooling = np.asarray(metadata["token_pooling"])
    offset, token_offset, result = 0, 0, []
    side = math.isqrt(patches_per_crop)
    if side * side != patches_per_crop:
        raise ValueError("Expected square ViT crop patch grid")
    for image_i, (grid, crops) in enumerate(zip(image_grids, image_num_crops)):
        mapping = np.asarray(metadata["subpatch_mapping"][image_i])
        h, w = mapping.shape
        patch_boxes = {}
        for row in range(side):
            for col in range(side):
                patch_boxes[offset + row * side + col] = (
                    col / side, row / side, (col + 1) / side, (row + 1) / side)
        for row in range(h):
            for col in range(w):
                patch_boxes[int(mapping[row, col])] = (col / w, row / h,
                                                       (col + 1) / w, (row + 1) / h)
        count = int(grid[0] * grid[1] + grid[2] * grid[3])
        for indices in pooling[token_offset:token_offset + count]:
            boxes = [patch_boxes[int(i)] for i in indices if i >= 0]
            if not boxes:
                raise ValueError("Pooled visual token has no valid support")
            result.append((image_i, min(b[0] for b in boxes), min(b[1] for b in boxes),
                           max(b[2] for b in boxes), max(b[3] for b in boxes)))
        token_offset += count
        offset += int(crops) * patches_per_crop
    if token_offset != len(pooling):
        raise ValueError("Unmapped visual tokens")
    return np.array(result, dtype=np.float32)


def roi_visible(boxes, roi, image_index=0):
    boxes = np.asarray(boxes)
    x0, y0, x1, y1 = roi
    return ((boxes[:, 0] == image_index) & (boxes[:, 1] < x1) & (boxes[:, 3] > x0)
            & (boxes[:, 2] < y1) & (boxes[:, 4] > y0))
