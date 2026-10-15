"""Evaluate a checkpoint on typed JSONL records with generic task metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import load_rows, normalize  # noqa: E402
from spatial_jev.kev import KevBuilder, build_model, make_processor, predict_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    config.update(preserve_option_order=True, max_image_tokens=1024, max_sequence_length=8192)
    raw = load_rows(args.data)
    requests = [normalize(row, args.data.parent, i) for i, row in enumerate(raw)]
    processor = make_processor(config)
    builder = KevBuilder(processor, config)
    import torch
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = build_model(config, args.checkpoint, trainable=False).to(device).eval()
    predictions = {}
    for start in range(0, len(requests), args.batch_size):
        batch = requests[start : start + args.batch_size]
        values = predict_batch(model, builder, batch, device)
        predictions.update({row["id"]: value for row, value in zip(batch, values, strict=True)})
    choice_correct, choice_total = 0, 0
    scalar_errors, point_errors = [], []
    for index, row in enumerate(raw):
        request = requests[index]
        target = row.get("target", {})
        prediction = predictions[request["id"]]["prediction"]
        kind = request["input"]["answer_space"]["kind"]
        if kind == "choice":
            choice_total += 1
            choice_correct += int(prediction == target.get("option_id"))
        elif kind == "scalar":
            scalar_errors.append(abs(float(prediction) - float(target["value"])))
        elif kind == "point":
            point = np.asarray(prediction, dtype=float)
            gold = np.asarray(target["points"][0], dtype=float)
            point_errors.append(float(np.linalg.norm(point - gold)))
    result = {
        "n": len(requests),
        "choice_accuracy": choice_correct / choice_total if choice_total else None,
        "scalar_mae": float(np.mean(scalar_errors)) if scalar_errors else None,
        "point_mean_distance": float(np.mean(point_errors)) if point_errors else None,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
