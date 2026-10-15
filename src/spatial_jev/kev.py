"""Multimodal KEV decisions using the existing spatial schemas and controllers.

Pointer formula and delimiters follow jaredpalmer/kev@f1535963 (Apache-2.0).
Each original QA is a private causal branch, including its successive crops.
"""

import hashlib
import json
import math
import random
import re
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from .hierarchy import ROOT_BOX, child_box, decode_point, decode_scalar_expectation, point_path, scalar_fine_bins
from .numeric_tasks import ANGLE_CODEBOOK, COUNT_CODEBOOK, codebook_for, decode_numeric, numeric_done, numeric_path
from .qwen35 import LORA_TARGETS, configure_full_language, sample_seed, train_mode
from .qwen35 import make_processor as make_processor  # noqa: PLC0414 -- explicit public re-export
from .roi_replay import (
    EMPTY_ROI_IMAGE,
    KV_REPLAY,
    REPLAY_VARIANTS,
    expand_replay_inputs,
    kv_replay_hidden,
)
from .unified import UnifiedBuilder
from .visual_token_pointing import (
    ORIGINAL_IMAGE,
    VISUAL_TOKENS,
    decode_visual_points,
    image_token_candidates,
    target_cells,
    validate_pointing_config,
)

UPSTREAM_REVISION = "f1535963cea021439370c23127bc970b6788e730"
SPECIAL = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
CONTROL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def escape(text):
    return CONTROL.sub(r"<¦\1¦>", text)


@lru_cache(maxsize=100000)
def _file_hash(path, size, mtime_ns):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def media_key(media):
    # Content identities, not image-pad token IDs or mutable paths, identify images.
    path = Path(media["uri"])
    stat = path.stat()
    return _file_hash(str(path), stat.st_size, stat.st_mtime_ns)


def state_key(inp):
    return tuple(media_key(m) for m in inp["media"])


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


class PointerHead(nn.Module):
    """KEV's affine query/key projections and scaled dot product, in FP32."""

    def __init__(self, hidden_size, pointer_dim=256):
        super().__init__()
        self.q = nn.Linear(hidden_size, pointer_dim)
        self.k = nn.Linear(hidden_size, pointer_dim)
        self.scale = 1 / math.sqrt(pointer_dim)

    def forward(self, decisions, options, owner, slot, counts):
        with torch.autocast(device_type=decisions.device.type, enabled=False):
            scores = (self.k(options.float()) * self.q(decisions.float())[owner]).sum(-1)
            scores = scores * self.scale
            logits = scores.new_full((len(decisions), int(counts.max())), -torch.inf)
            return logits.index_put((owner, slot), scores)


class KevBuilder(UnifiedBuilder):
    def __init__(self, processor, config):
        validate_pointing_config(config)
        if (config.get("numeric_codebooks") is not None
                and config["numeric_codebooks"] != {"count": COUNT_CODEBOOK, "deg": ANGLE_CODEBOOK}):
            raise ValueError("Checkpoint numeric codebooks differ from this implementation")
        self.processor, self.config = processor, config
        self.codebook, self.variant = config["scalar_codebook"], config.get("point_variant", "crop_refill")
        if self.codebook.get("fine_bins_by_root") is not None:
            for root in range(1, len(self.codebook["edges_m"])):
                if scalar_fine_bins(root, self.codebook) > config["max_choices"]:
                    raise ValueError("Scalar refinement exceeds model candidate budget")
        self.visual_token_pointing = self.variant == VISUAL_TOKENS
        self.markers = [processor.tokenizer.convert_tokens_to_ids(t) for t in SPECIAL]
        if len(set(self.markers)) != 5:
            raise ValueError("KEV requires five distinct existing delimiter tokens")

    def stage(self, inp, seed, box=ROOT_BOX, level=0, scalar_root=None):
        task = inp["answer_space"]["kind"]
        if task == "point" and self.visual_token_pointing:
            instruction = f"Request: {inp['question']}\nLocate the requested target in the image."
            # No artificial region options: the candidates are image-token positions.
            return SPECIAL[1] + escape(instruction) + SPECIAL[4], 0, []
        mapping = []
        if task == "choice":
            options = list(inp["answer_space"]["options"])
            if not (self.config.get("preserve_option_order", False)
                    or inp["answer_space"].get("preserve_option_order", False)):
                random.Random(seed).shuffle(options)
            mapping = [o["id"] for o in options]
            instruction = f"Question: {inp['question']}\nSelect exactly one option."
            options = [o["text"] for o in options]
        else:
            # Use the existing authoritative spatial prompts, including interval semantics.
            prompt = (
                self.point_prompt(inp, box, level)
                if task == "point"
                else self.scalar_prompt(inp, scalar_root)[0]
            )
            instruction, tail = prompt.rsplit("\nOptions:\n", 1)
            option_text, suffix = tail.rsplit("\nSelect exactly one option.", 1)
            if suffix:
                raise ValueError("Unexpected spatial prompt suffix")
            options = [line.split(". ", 1)[1] for line in option_text.splitlines()]
            instruction += "\nSelect exactly one option."
        text = SPECIAL[1] + escape(instruction)
        text += "".join(SPECIAL[2] + escape(o) + SPECIAL[3] for o in options)
        return text + SPECIAL[4], len(options), mapping

    def initial_spec(self, inp, identity):
        if self.variant in (VISUAL_TOKENS, ORIGINAL_IMAGE, *REPLAY_VARIANTS) and inp["answer_space"]["kind"] == "point":
            if len(inp["media"]) != 1 or inp["media"][0].get("kind", "image") != "image":
                raise ValueError("Original-image / visual-token pointing requires exactly one still image")
            if inp["answer_space"].get("coordinate_system", "normalized_xy") != "normalized_xy":
                raise ValueError("Original-image / visual-token pointing requires normalized_xy coordinates")
        text, count, mapping = self.stage(inp, sample_seed(identity, self.config["seed"]))
        keys = state_key(inp)
        return {
            "text": SPECIAL[0] + IMAGE * len(keys) + text,
            "images": self.load_images(inp),
            "image_keys": list(keys),
            "state_key": keys,
            "sample_id": identity,
            "counts": [count],
            "mapping": mapping,
            "box": ROOT_BOX,
            "path": [],
            "input": inp,
        }

    def advance(self, spec, choice):
        inp, path = spec["input"], spec["path"]
        task = inp["answer_space"]["kind"]
        if task == "point" and self.visual_token_pointing:
            raise ValueError("Visual-token pointing does not advance or crop")
        extra = ""
        if task == "point":
            spec["box"] = child_box(spec["box"], choice)
            if self.variant in REPLAY_VARIANTS:
                spec.setdefault('roi_boxes', []).append(spec['box'])
                extra = EMPTY_ROI_IMAGE
            elif self.variant != ORIGINAL_IMAGE:
                spec["images"].append(self.crop(spec["images"][0], spec["box"]))
                spec["image_keys"].append(spec["state_key"][0] + ":crop:" + repr(spec["box"]))
                extra = IMAGE
        text, count, _ = self.stage(
            inp,
            self.config["seed"],
            spec["box"],
            len(path),
            scalar_root=path[0] if task == "scalar" else None,
        )
        spec["text"] += f"Selected option: {choice + 1}." + extra + text
        spec["counts"].append(count)

    def training_spec(self, row):
        spec = self.initial_spec(row["input"], row["sample_id"])
        kind = row["input"]["answer_space"]["kind"]
        target = row["target"]
        if kind == "point" and self.visual_token_pointing:
            # Labels stay out of text/images; cell indices depend on the processor's grid.
            spec.update(target_points=target["points"], weight=row.get("_weight", 1.0))
            return spec
        path = (
            [spec["mapping"].index(target["option_id"])]
            if kind == "choice"
            else numeric_path(target["value"], codebook_for(row["input"], self.codebook))
            if kind == "scalar"
            else point_path(target["points"][0])
        )
        for choice in path[:-1]:
            spec["path"].append(choice)
            self.advance(spec, choice)
        spec.update(targets=path, weight=row.get("_weight", 1.0))
        return spec

    def encode_specs(self, specs, training=True):
        encoded = dict(
            self.processor(
                text=[s["text"] for s in specs],
                images=[im for s in specs for im in s["images"]],
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
                device="cpu",
            )
        )
        encoded.pop("token_type_ids", None)
        if self.variant in REPLAY_VARIANTS:
            encoded = expand_replay_inputs(self, specs, encoded)
        lengths = encoded["attention_mask"].sum(-1).tolist()
        if max(lengths) > self.config["max_sequence_length"]:
            raise ValueError(f"KEV sequence budget exceeded without truncation: {lengths}")
        if "mm_token_type_ids" not in encoded:
            encoded["mm_token_type_ids"] = (
                encoded["input_ids"]
                == self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            ).long()
        positions, option_positions, owner, slot, counts, targets, weights = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        prefix_lengths = []
        image_cursor = 0
        visual_owners, visual_slots, visual_mass = [], [], []
        has_visual = False
        for row, spec in enumerate(specs):
            ids = encoded["input_ids"][row, : lengths[row]].tolist()
            prefix_lengths.append(ids.index(self.markers[1]))
            ends = [i for i, t in enumerate(ids) if t == self.markers[4]]
            opt_ends = [i for i, t in enumerate(ids) if t == self.markers[3]]
            first_grid = encoded["image_grid_thw"][image_cursor]
            image_cursor += len(spec["images"])
            if self.visual_token_pointing and spec["input"]["answer_space"]["kind"] == "point":
                if len(ends) != 1 or opt_ends or len(spec["images"]) != 1:
                    raise ValueError("Visual pointing must have one query, one image and no text candidates")
                candidates, shape = image_token_candidates(
                    ids, self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                    first_grid.tolist(), self.processor.image_processor.merge_size,
                )
                if candidates[-1] >= prefix_lengths[-1] or prefix_lengths[-1] >= ends[0]:
                    raise ValueError("Visual keys must precede the private question and query")
                decision = len(positions)
                count = len(candidates)
                positions.append((row, ends[0]))
                option_positions.extend((row, p) for p in candidates)
                owner.extend([decision] * count)
                slot.extend(range(count))
                counts.append(count)
                spec.update(counts=[count], visual_grid=shape)
                has_visual = True
                if training:
                    cells = target_cells(spec["target_points"], shape)
                    targets.append(cells[0])
                    weights.append(spec.get("weight", 1.0))
                    visual_owners.extend([decision] * len(cells))
                    visual_slots.extend(cells)
                    visual_mass.extend([1 / len(cells)] * len(cells))
                continue
            if len(ends) != len(spec["counts"]) or len(opt_ends) != sum(spec["counts"]):
                raise ValueError(f"Decision/candidate markers mismatch: {spec['sample_id']}")
            cursor = 0
            for stage, (pos, count) in enumerate(zip(ends, spec["counts"])):
                options = opt_ends[cursor : cursor + count]
                cursor += count
                if not all((ends[stage - 1] if stage else -1) < p < pos for p in options):
                    raise ValueError("Candidate belongs to a different decision stage")
                if not training and stage != len(ends) - 1:
                    continue
                owner.extend([len(positions)] * count)
                slot.extend(range(count))
                positions.append((row, pos))
                option_positions.extend((row, p) for p in options)
                counts.append(count)
                if training:
                    targets.append(spec["targets"][stage])
                    weights.append(spec.get("weight", 1.0) / len(ends))
        # One frozen vision forward per unique image/crop in this batch. Keep full grids
        # for each row's native M-RoPE, and an explicit map back to each image occurrence.
        keys = [key for s in specs for key in s["image_keys"]]
        chunks = encoded["pixel_values"].split(encoded["image_grid_thw"].prod(-1).tolist())
        unique, grids, pixels, image_owner = {}, [], [], []
        for occurrence, (key, grid, pixel) in enumerate(
            zip(keys, encoded["image_grid_thw"], chunks, strict=True)
        ):
            signature = (key, tuple(grid.tolist()))
            if not self.config.get("deduplicate_vision", True):
                signature = (*signature, occurrence)
            if signature not in unique:
                unique[signature] = len(grids)
                grids.append(grid)
                pixels.append(pixel)
            image_owner.append(unique[signature])
        encoded["pixel_values"] = torch.cat(pixels)
        encoded["unique_grid_thw"] = torch.stack(grids)
        encoded["image_owner"] = torch.tensor(image_owner)
        packed = {
            "inputs": encoded,
            "positions": torch.tensor(positions),
            "option_positions": torch.tensor(option_positions),
            "owner": torch.tensor(owner),
            "slot": torch.tensor(slot),
            "counts": torch.tensor(counts),
            "targets": torch.tensor(targets),
            "weights": torch.tensor(weights),
            "prefix_lengths": torch.tensor(prefix_lengths),
            "lengths": torch.tensor(lengths),
            "state_keys": [s["state_key"] for s in specs],
            "image_keys": list(unique),
            "qa_count": sum(s.get("weight", 1.0) for s in specs),
            "tokens": sum(lengths),
            "padded_tokens": encoded["input_ids"].numel(),
        }
        if self.variant == KV_REPLAY and 'roi_kv_sources' in encoded:
            packed['roi_kv_copy'] = True
        if has_visual:
            packed["visual_pointing"] = True
            if training:
                packed.update(visual_target_owner=torch.tensor(visual_owners, dtype=torch.long),
                              visual_target_slot=torch.tensor(visual_slots, dtype=torch.long),
                              visual_target_mass=torch.tensor(visual_mass, dtype=torch.float32))
        return packed

    def collate(self, rows):
        return self.encode_specs([self.training_spec(row) for row in rows])


class KevSpatialModel(nn.Module):
    def __init__(self, backbone, pointer):
        super().__init__()
        self.backbone, self.pointer = backbone, pointer
        self.language_fp32 = False

    @property
    def native(self):
        return (
            self.backbone.get_base_model()
            if hasattr(self.backbone, "get_base_model")
            else self.backbone
        )

    def train(self, mode=True):
        super().train(mode)
        self.native.visual.eval()
        return self

    def language_context(self, device_type):
        return (
            torch.autocast(device_type=device_type, enabled=False)
            if self.language_fp32
            else nullcontext()
        )

    def embeddings(self, inputs, features=None):
        native = self.native
        hidden = native.get_input_embeddings()(inputs["input_ids"])
        if features is None:
            with torch.no_grad():
                features = native.get_image_features(
                    inputs["pixel_values"], inputs["unique_grid_thw"], return_dict=True
                ).pooler_output
        features = torch.cat([features[i] for i in inputs["image_owner"].tolist()]).to(hidden.dtype)
        if 'roi_feature_sources' in inputs:
            features = features.index_select(0, inputs['roi_feature_sources'])
        mask, _ = native.get_placeholder_mask(inputs["input_ids"], hidden, image_features=features)
        hidden = hidden.masked_scatter(mask, features)
        positions, _ = native.get_rope_index(
            inputs.get('roi_base_ids', inputs["input_ids"]),
            inputs.get('roi_base_types', inputs["mm_token_type_ids"]),
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs.get('roi_base_mask', inputs["attention_mask"]),
        )
        if 'roi_position_sources' in inputs:
            positions = positions.gather(2, inputs['roi_position_sources'][None].expand(3, -1, -1))
        return hidden, positions

    def readout(self, hidden, packed, branch_only=False):
        if self.pointer is None:
            raise ValueError("Native QK requires multimodal positions; use model.forward")
        p, o = packed["positions"], packed["option_positions"]
        if branch_only:
            p = p.clone()
            o = o.clone()
            p[:, 1] -= packed["prefix_lengths"][p[:, 0]]
            o[:, 1] -= packed["prefix_lengths"][o[:, 0]]
        return self.pointer(
            hidden[p[:, 0], p[:, 1]],
            hidden[o[:, 0], o[:, 1]],
            packed["owner"],
            packed["slot"],
            packed["counts"],
        )

    def forward(self, packed, shared_prefix=False):
        if packed.get("visual_pointing", False) and shared_prefix:
            raise ValueError("Visual-token keys are in the prefix; use the full causal-row forward")
        if self.pointer is None and shared_prefix:
            raise ValueError("Native QK currently requires independent causal rows")
        hidden, positions = self.embeddings(packed["inputs"])
        with self.language_context(hidden.device.type):
            if self.pointer is None and getattr(self.native.language_model.config, "kev_readout_layer_indices", None):
                from .kev_multilayer_qk import multilayer_forward

                return multilayer_forward(self.native.language_model, hidden, positions,
                                          packed["inputs"]["attention_mask"], packed)
            if packed.get('roi_kv_copy', False):
                hidden = kv_replay_hidden(self.native.language_model, hidden, positions,
                                         packed['inputs']['attention_mask'], packed['inputs']['roi_kv_sources'])
            elif shared_prefix:
                from .kev_prefix import shared_hidden

                hidden = shared_hidden(self.native.language_model, hidden, positions, packed)
            else:
                hidden = self.native.language_model(
                    inputs_embeds=hidden,
                    position_ids=positions,
                    attention_mask=packed["inputs"]["attention_mask"],
                    use_cache=False,
                ).last_hidden_state
            if self.pointer is None:
                lm = self.native.language_model
                rope_positions = positions[1:] if positions.shape[0] == 4 else positions
                return lm.native_qk(hidden, packed, lm.rotary_emb(hidden, rope_positions))
        return self.readout(hidden, packed, branch_only=shared_prefix)


def build_model(config, checkpoint=None, trainable=True):
    from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5TextModel

    validate_pointing_config(config)
    mode = train_mode(config)
    kind = config.get("readout_kind", "added_pointer")
    if kind not in (
        "added_pointer",
        "native_last_qk_mean",
        "native_merged_qk256",
        "native_last_qk_concat",
    ):
        raise ValueError(f"Unsupported KEV readout: {kind}")
    native_qk = kind != "added_pointer"
    fusion_layers = config.get("readout_layer_indices")
    if fusion_layers is not None and (kind != "native_last_qk_mean"
            or config.get("layer_fusion") != "equal_logits" or config.get("gradient_checkpointing", False)
            or config.get("point_variant", "crop_refill") != "crop_refill"):
        raise ValueError("Multilayer ablation fixes mean heads, equal logits, old pointing and no gradient checkpointing")
    if checkpoint:
        saved = json.loads((Path(checkpoint) / "config.json").read_text())
        if train_mode(saved) != mode:
            raise ValueError("Cannot load a KEV checkpoint with a different training mode")
        if saved.get("readout_kind", "added_pointer") != kind:
            raise ValueError("Cannot load a KEV checkpoint with a different readout")
        if saved.get("readout_layer_indices") != fusion_layers or saved.get("layer_fusion") != config.get("layer_fusion"):
            raise ValueError("Cannot load a checkpoint with a different multilayer readout")
        if saved.get("point_variant", "crop_refill") != config.get("point_variant", "crop_refill"):
            raise ValueError("Cannot load a checkpoint with a different pointing method")
    if native_qk and (mode != "full_language" or config.get("shared_prefix", False)):
        raise ValueError("Native QK requires full_language and independent causal rows")
    full = Qwen3_5ForConditionalGeneration.from_pretrained(
        config["base_model"],
        local_files_only=config.get("local_files_only", False),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    backbone, hidden_size = full.model, full.config.text_config.hidden_size
    del full
    backbone.requires_grad_(False)
    if mode == "full_language":
        if checkpoint:
            directory = Path(checkpoint) / "language_model"
            text_config = Qwen3_5TextConfig.from_pretrained(
                directory, local_files_only=config.get("local_files_only", False)
            )
            text_class = Qwen3_5TextModel
            if native_qk:
                if getattr(text_config, "kev_readout_kind", None) != kind:
                    raise ValueError("Language checkpoint readout marker disagrees with config")
                if (getattr(text_config, "kev_readout_layer_indices", None) != fusion_layers
                        or getattr(text_config, "kev_layer_fusion", None) != config.get("layer_fusion")):
                    raise ValueError("Language checkpoint multilayer markers disagree with config")
                if getattr(text_config, "kev_readout_fp32", False) != config.get(
                    "native_readout_fp32", False
                ):
                    raise ValueError("Language checkpoint readout precision disagrees with config")
                expected_layer = config.get(
                    "readout_layer_index", text_config.num_hidden_layers - 1
                )
                if (
                    getattr(
                        text_config, "kev_readout_layer_index", text_config.num_hidden_layers - 1
                    )
                    != expected_layer
                ):
                    raise ValueError("Language checkpoint readout layer disagrees with config")
                from .kev_native_qk import NativeQKTextModel

                text_class = NativeQKTextModel
            backbone.language_model, loading = text_class.from_pretrained(
                directory,
                config=text_config,
                local_files_only=config.get("local_files_only", False),
                dtype=torch.float32,
                attn_implementation="sdpa",
                output_loading_info=True,
            )
            if any(
                loading.get(k)
                for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            ):
                raise ValueError(f"Incomplete language checkpoint load: {loading}")
            backbone.config.text_config = backbone.language_model.config
        elif native_qk:
            from .kev_native_qk import prune_native_qk

            prune_native_qk(backbone.language_model, kind, config.get("readout_layer_index"))
        configure_full_language(backbone, trainable=trainable)
        if native_qk:
            if fusion_layers is not None:
                from .kev_multilayer_qk import configure_multilayer

                configure_multilayer(backbone.language_model, fusion_layers)
            backbone.language_model.native_qk.readout_fp32 = config.get(
                "native_readout_fp32", False
            )
            backbone.language_model.config.kev_readout_fp32 = config.get(
                "native_readout_fp32", False
            )
    else:
        from peft import LoraConfig, PeftModel, get_peft_model

        if checkpoint:
            backbone = PeftModel.from_pretrained(
                backbone, str(Path(checkpoint) / "adapter"), is_trainable=trainable
            )
        else:
            backbone = get_peft_model(
                backbone,
                LoraConfig(
                    r=config["lora_rank"],
                    lora_alpha=config["lora_alpha"],
                    lora_dropout=config["lora_dropout"],
                    target_modules=LORA_TARGETS,
                    bias="none",
                ),
            )
    from .training_alignment import seed_head

    pointer = (
        None
        if native_qk
        else seed_head(config, lambda: PointerHead(hidden_size, config.get("pointer_dim", 256)))
    )
    if checkpoint and pointer is not None:
        from safetensors.torch import load_file

        pointer.load_state_dict(load_file(str(Path(checkpoint) / "pointer.safetensors")))
    if trainable and config.get("gradient_checkpointing", False):
        native = backbone.get_base_model() if mode == "lora" else backbone
        native.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = KevSpatialModel(backbone, pointer)
    inference_dtype = config.get("inference_language_dtype", "bf16")
    if inference_dtype not in ("bf16", "fp32"):
        raise ValueError("inference_language_dtype must be bf16 or fp32")
    if inference_dtype == "fp32":
        if trainable:
            raise ValueError("FP32 language compute is an inference verification option")
        model.native.language_model.float()
        model.language_fp32 = True
    if any(p.requires_grad for p in model.native.visual.parameters()):
        raise ValueError("Vision and merger must remain frozen")
    if trainable and mode == "full_language":
        if not all(
            p.requires_grad and p.dtype == torch.float32
            for p in model.native.language_model.parameters()
        ):
            raise ValueError("Full language training requires every language parameter in FP32")
        if any("lora_" in n for n, _ in model.named_parameters()):
            raise ValueError("Full language training must not contain LoRA adapters")
    if not trainable:
        model.requires_grad_(False).eval()
    return model


def save_checkpoint(model, processor, config, path, step):
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    mode = train_mode(config)
    if mode == "full_language":
        model.native.language_model.save_pretrained(
            path / "language_model", safe_serialization=True, max_shard_size="4GB"
        )
    else:
        model.backbone.save_pretrained(path / "adapter", safe_serialization=True)
    if model.pointer is not None:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.pointer.state_dict().items()},
            str(path / "pointer.safetensors"),
        )
    else:
        metadata_path = path / "language_model/config.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["architectures"] = ["NativeQKTextModel"]
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    processor.save_pretrained(path / "processor")
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (path / "checkpoint.json").write_text(
        json.dumps(
            {
                "step": step,
                "format": "kev-spatial/v1",
                "train_mode": mode,
                "readout_kind": config.get("readout_kind", "added_pointer"),
                "point_variant": config.get("point_variant", "crop_refill"),
                "readout_layer_index": getattr(
                    model.native.language_model.config, "kev_readout_layer_index", None
                ),
                "readout_layer_indices": config.get("readout_layer_indices"),
                "layer_fusion": config.get("layer_fusion"),
                "checkpoint_policy": config.get("checkpoint_policy", "full"),
                "upstream_revision": UPSTREAM_REVISION,
                "base_revision": config["base_revision"],
            },
            indent=2,
        )
        + "\n"
    )


def validate_checkpoint_parameters(model, config, checkpoint):
    """Compare saved tensors shard by shard without loading a second full model."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    if train_mode(config) == "full_language":
        loaded = model.native.language_model.state_dict()
        seen = set()
        for shard in sorted((checkpoint / "language_model").glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as stream:
                for name in stream.keys():  # noqa: SIM118 -- safetensors reader is not a mapping
                    if name in seen or name not in loaded:
                        raise ValueError(f"Unexpected checkpoint parameter: {name}")
                    tensor = stream.get_tensor(name)
                    if not torch.equal(tensor, loaded[name].detach().cpu()):
                        raise ValueError(f"Language checkpoint round trip differs: {name}")
                    seen.add(name)
        if seen != loaded.keys():
            raise ValueError("Language checkpoint is missing state tensors")
    else:
        from peft import get_peft_model_state_dict

        saved = load_file(str(checkpoint / "adapter/adapter_model.safetensors"))
        loaded = get_peft_model_state_dict(model.backbone)
        if saved.keys() != loaded.keys() or any(
            not torch.equal(saved[k], loaded[k].detach().cpu()) for k in saved
        ):
            raise ValueError("Adapter checkpoint round trip is not exact")
    if model.pointer is None:
        if (checkpoint / "pointer.safetensors").exists():
            raise ValueError("Native QK checkpoint must not contain an added pointer")
        return
    head = load_file(str(checkpoint / "pointer.safetensors"))
    loaded_head = model.pointer.state_dict()
    if head.keys() != loaded_head.keys() or any(
        not torch.equal(v, loaded_head[k].detach().cpu()) for k, v in head.items()
    ):
        raise ValueError("Pointer checkpoint round trip is not exact")


@torch.inference_mode()
def predict_batch(model, builder, requests, device, runtime=None, point_decoding=None, scalar_decoding="argmax"):
    """Free rollout: only requests (no targets), with a private crop history per QA."""
    visual = getattr(builder, "visual_token_pointing", False)
    if scalar_decoding not in ("argmax", "fine_expectation"):
        raise ValueError("Unknown scalar decoding mode")
    if point_decoding is not None and (not visual or point_decoding not in ("argmax", "components")):
        raise ValueError("point_decoding overrides require visual-token pointing")
    if visual and runtime is not None:
        raise ValueError("Visual-token pointing currently uses the uncached causal-row forward")
    states, results = [], [None] * len(requests)
    for index, request in enumerate(requests):
        spec = builder.initial_spec(request["input"], request["id"])
        spec.update(index=index, logits=[])
        states.append(spec)
    while states:
        packed = to_device(builder.encode_specs(states, training=False), device)
        logits = model(packed) if runtime is None else runtime(packed)
        choices = logits.argmax(-1).tolist()
        following = []
        for state, choice, scores in zip(states, choices, logits.float().cpu()):
            path, inp = state["path"], state["input"]
            if visual and inp["answer_space"]["kind"] == "point":
                values = scores[:state["counts"][-1]].tolist()
                results[state["index"]] = {
                    **decode_visual_points(values, state["visual_grid"],
                        decoder=point_decoding or builder.config.get("point_decoder", "argmax"),
                        relative_threshold=builder.config.get("point_component_threshold", 0.1)),
                    "path": [], "logits": [values],
                }
                continue
            path.append(choice)
            state["logits"].append(scores[: state["counts"][-1]].tolist())
            kind = inp["answer_space"]["kind"]
            done = (
                kind == "choice"
                or (kind == "point" and len(path) == 3)
                or (
                    kind == "scalar"
                    and numeric_done(path, codebook_for(inp, builder.codebook))
                )
            )
            if done:
                prediction = (
                    state["mapping"][choice]
                    if kind == "choice"
                    else decode_point(path)
                    if kind == "point"
                    else decode_numeric(path, codebook_for(inp, builder.codebook))
                )
                if (kind == "scalar" and inp["answer_space"].get("unit", "m") == "m"
                        and scalar_decoding == "fine_expectation"):
                    prediction = decode_scalar_expectation(
                        path[0], state["logits"][-1] if len(path) == 2 else [], builder.codebook)
                results[state["index"]] = {
                    "prediction": prediction,
                    "path": path,
                    "logits": state["logits"],
                }
            else:
                builder.advance(state, choice)
                following.append(state)
        states = following
    return results
