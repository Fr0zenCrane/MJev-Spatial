"""Nine-way conditional trees and connected-component multi-point readout."""

import numpy as np

from .hierarchy import POINT_RC


def softmax9(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("Expected nine finite logits")
    probabilities = np.exp(values - values.max())
    return probabilities / probabilities.sum()


def grid_location(path):
    row = col = 0
    for choice in path:
        dr, dc = POINT_RC[choice]
        row, col = row * 3 + dr, col * 3 + dc
    return row, col


def branch_spec(builder, parent, choice):
    # Copy mutable histories, but share immutable PIL images and input metadata.
    child = dict(parent)
    for key in ("images", "image_keys", "counts", "mapping", "path"):
        child[key] = list(parent[key])
    if 'roi_boxes' in parent:
        child['roi_boxes'] = list(parent['roi_boxes'])
    child["path"].append(choice)
    builder.advance(child, choice)
    return child


def probability_map(builder, request, evaluate_nodes, mode="full", root_relative_cutoff=0.01):
    """Three levels, 729 leaves; labels/masks/counts are not accepted as arguments.

    root_pruned uses only the first decision to omit low-mass root subtrees.
    Unvisited leaves remain explicitly unknown, with their missing mass reported.
    No probability renormalization is applied after pruning.
    """
    if mode not in ("full", "root_pruned"):
        raise ValueError("Unknown tree policy")
    if not 0 <= root_relative_cutoff <= 1:
        raise ValueError("Invalid root cutoff")
    if request["input"]["answer_space"]["kind"] != "point":
        raise ValueError("Point input required")
    grid = np.zeros((27, 27), dtype=np.float64)
    known = np.zeros((27, 27), dtype=bool)
    nodes, pruned = [], []
    frontier = [(builder.initial_spec(request["input"], request["id"]), 1.0)]
    for depth in range(3):
        logits = evaluate_nodes([state for state, _ in frontier])
        if len(logits) != len(frontier):
            raise ValueError("Node evaluator returned a different batch size")
        following = []
        for (state, mass), scores in zip(frontier, logits):
            probabilities = softmax9(scores)
            path = list(state["path"])
            nodes.append({"path": path, "mass": mass, "logits": list(map(float, scores)),
                          "probabilities": probabilities.tolist()})
            cutoff = probabilities.max() * root_relative_cutoff
            for choice, probability in enumerate(probabilities):
                child_path = [*path, choice]
                child_mass = mass * float(probability)
                if depth == 2:
                    row, col = grid_location(child_path)
                    if known[row, col]:
                        raise ValueError("Duplicate leaf location")
                    grid[row, col], known[row, col] = child_mass, True
                elif depth == 0 and mode == "root_pruned" and probability < cutoff:
                    pruned.append({"path": child_path, "mass": child_mass})
                else:
                    following.append((branch_spec(builder, state, choice), child_mass))
        frontier = following
    retained, omitted = float(grid.sum()), sum(r["mass"] for r in pruned)
    if not np.isclose(retained + omitted, 1, atol=1e-12, rtol=0):
        raise ValueError("Probability mass was lost or duplicated")
    return {"probabilities": grid, "known": known, "nodes": nodes,
            "evaluated_nodes": len(nodes), "evaluated_leaves": int(known.sum()),
            "retained_mass": retained, "pruned_mass": omitted, "pruned_branches": pruned}


def component_points(probabilities, known=None, relative_threshold=0.1, connectivity=8):
    """Pick each superlevel component's maximum cell, with deterministic tie breaks."""
    grid = np.asarray(probabilities, dtype=np.float64)
    if grid.ndim != 2 or not np.isfinite(grid).all() or (grid < 0).any():
        raise ValueError("Invalid spatial probability grid")
    if not 0 < relative_threshold <= 1 or connectivity not in (4, 8):
        raise ValueError("Invalid connected-component settings")
    known = np.ones_like(grid, dtype=bool) if known is None else np.asarray(known, dtype=bool)
    if known.shape != grid.shape:
        raise ValueError("Known-cell mask shape mismatch")
    peak = float(grid[known].max()) if known.any() else 0.0
    threshold = peak * relative_threshold
    mask = known & (grid >= threshold) & (grid > 0)
    visited = np.zeros_like(mask)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    height, width = grid.shape
    components, points = [], []
    for row, col in zip(*np.where(mask)):
        if visited[row, col]:
            continue
        visited[row, col] = True
        stack, cells = [(int(row), int(col))], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in neighbors:
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        # Prefer the highest probability, then topmost/leftmost on exact ties.
        y, x = min(cells, key=lambda cell: (-grid[cell], cell[0], cell[1]))
        point = [(x + 0.5) / width, (y + 0.5) / height]
        points.append(point)
        components.append({"cells": [list(cell) for cell in cells], "peak_cell": [y, x],
                           "peak_probability": float(grid[y, x]),
                           "mass": sum(float(grid[cell]) for cell in cells)})
    return {"points": points, "components": components, "threshold": threshold,
            "relative_threshold": relative_threshold, "connectivity": connectivity}
