"""Run MJev-Spatial inference on a JSON or JSONL request file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_jev.kev import KevBuilder, build_model, make_processor, predict_batch, to_device


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [json.loads(path.read_text())]


def normalize(row: dict, request_dir: Path, index: int) -> dict:
    value = row.get("input", row)
    value = json.loads(json.dumps(value))
    for media in value.get("media", []):
        uri = Path(media["uri"])
        if not uri.is_absolute():
            media["uri"] = str((request_dir / uri).resolve())
    identity = str(row.get("id", row.get("sample_id", f"request-{index:06d}")))
    return {"id": identity, "input": value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--language-fp32", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    config = json.loads((args.checkpoint / "config.json").read_text())
    config.update(
        preserve_option_order=True,
        max_image_tokens=1024,
        max_sequence_length=max(8192, int(config.get("max_sequence_length", 4096))),
        inference_language_dtype="fp32" if args.language_fp32 else "bf16",
    )
    requests = [normalize(row, args.input.parent, i) for i, row in enumerate(load_rows(args.input))]
    if len({row["id"] for row in requests}) != len(requests):
        raise ValueError("Request IDs must be unique")
    processor = make_processor(config)
    builder = KevBuilder(processor, config)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = build_model(config, args.checkpoint, trainable=False).to(device).eval()
    results = []
    for start in range(0, len(requests), args.batch_size):
        batch = requests[start : start + args.batch_size]
        with torch.inference_mode():
            values = predict_batch(model, builder, batch, device)
        results.extend({"id": row["id"], **value} for row, value in zip(batch, values, strict=True))
    text = "".join(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n" for value in results)
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
