"""Shared controls for the three full-language architecture experiments."""

import hashlib
import json
import math

import torch


def old_head_factors(config, step, total_steps):
    """Exactly the old runner's zero-based, pre-optimizer-step LR multipliers."""
    decay = 0.1 + 0.9 * (1 + math.cos(math.pi * step / total_steps)) / 2
    warm = min(1, (step + 1) / config["warmup_steps"])
    backbone_warm = min(1, max(0, step + 1 - config["head_warmup_steps"]) / config["warmup_steps"])
    return backbone_warm * decay, warm * decay


def seed_head(config, factory):
    if "head_seed" not in config:
        return factory()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["head_seed"])
        return factory()


def state_hash(module):
    return hashlib.sha256(
        b"".join(
            value.detach().float().cpu().contiguous().numpy().tobytes()
            for value in module.state_dict().values()
        )
    ).hexdigest()


def canonical_rows(rows, path, expected_sha256):
    """Restore the ORIGINAL flat order before shuffling, including padding row 0."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Canonical training file hash changed")
    by_id = {row["sample_id"]: row for row in rows}
    identities = []
    with path.open() as stream:
        for line in stream:
            source = json.loads(line)
            identity = source["sample_id"]
            identities.append(identity)
            if identity not in by_id or any(
                source.get(key) != by_id[identity].get(key) for key in ("input", "target")
            ):
                raise ValueError(f"Canonical QA contents differ: {identity}")
    if (
        len(by_id) != len(rows)
        or len(set(identities)) != len(identities)
        or set(identities) != by_id.keys()
    ):
        raise ValueError("Canonical QA identities do not match the training dataset")
    return [by_id[identity] for identity in identities]
