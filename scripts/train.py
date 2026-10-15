"""Continue training an MJev-Spatial checkpoint on already-labeled JSONL data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_jev.kev import KevBuilder, build_model, make_processor, save_checkpoint, to_device  # noqa: E402
from spatial_jev.visual_token_pointing import decision_losses  # noqa: E402


class Records(Dataset):
    def __init__(self, path: Path):
        self.rows = []
        for index, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            raw = json.loads(line)
            value = json.loads(json.dumps(raw.get("input", raw)))
            for media in value.get("media", []):
                uri = Path(media["uri"])
                if not uri.is_absolute():
                    media["uri"] = str((path.parent / uri).resolve())
            self.rows.append({
                "schema_version": "spatial-decision/v3",
                "sample_id": str(raw.get("id", raw.get("sample_id", f"row-{index:06d}"))),
                "split": "train",
                "input": value,
                "target": raw["target"],
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def read_settings(path: Path) -> dict:
    if path.suffix == ".json":
        return json.loads(path.read_text())
    values = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if value.lower() in ("true", "false"):
            values[key] = value.lower() == "true"
        else:
            try:
                values[key] = float(value) if any(c in value for c in ".eE") else int(value)
            except ValueError:
                values[key] = value.strip("\"'")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    settings = read_settings(args.config)
    config = json.loads((args.checkpoint / "config.json").read_text())
    config.update(
        preserve_option_order=True,
        max_image_tokens=int(settings.get("max_image_tokens", 512)),
        max_sequence_length=int(settings.get("max_sequence_length", 4096)),
    )
    data = Records(args.data)
    if not data:
        raise ValueError("Training data is empty")
    processor = make_processor(config)
    builder = KevBuilder(processor, config)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = build_model(config, args.checkpoint, trainable=True).to(device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(settings.get("learning_rate", 5e-6)),
        weight_decay=float(settings.get("weight_decay", 0.01)),
    )
    loader = DataLoader(
        data,
        batch_size=int(settings.get("batch_size", 1)),
        shuffle=True,
        collate_fn=builder.collate,
    )
    max_steps = int(settings.get("max_steps", 3))
    accumulation = int(settings.get("gradient_accumulation_steps", 1))
    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        for packed in loader:
            packed = to_device(packed, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(packed)
                losses = decision_losses(logits, packed)
                loss = (losses * packed["weights"].to(device)).sum() / max(float(packed["qa_count"]), 1.0)
                loss = loss / accumulation
            loss.backward()
            if (step + 1) % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            if step >= max_steps:
                break
    args.output.mkdir(parents=True, exist_ok=False)
    save_checkpoint(model, processor, config, args.output, step)
    print(json.dumps({"status": "complete", "steps": step, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
