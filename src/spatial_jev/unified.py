"""One shared classifier with scalar interval decisions and three-step pointing."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .hierarchy import (POINT_NAMES, ROOT_BOX, child_box, decode_point, decode_scalar,
                        point_path, roi_visible, scalar_fine_bins, scalar_interval, scalar_path,
                        visual_token_boxes)
from .schema import compile_prompt
from .numeric_tasks import codebook_for, numeric_prompt


def causal_visual_mask(length, visual_types, stages, query_start=0):
    """Causal text plus bidirectional vision ONLY within the same reveal stage.

    Native Molmo2's all-image bidirectionality would leak future GT-selected crops.
    Old prefix queries are not recomputed during incremental inference.
    """
    keys = torch.arange(length)
    queries = keys[query_start:]
    allowed = keys[None] <= queries[:, None]
    visual_types = torch.as_tensor(visual_types, dtype=torch.bool)
    stages = torch.as_tensor(stages, dtype=torch.long)
    allowed |= (visual_types[query_start:, None] & visual_types[None]
                & (stages[query_start:, None] == stages[None]))
    return torch.zeros((1, 1, len(queries), length)).masked_fill(~allowed[None, None], -torch.inf)


def apply_roi_mask(mask, visual_positions, boxes, roi, start=0, end=None):
    keep = roi_visible(boxes, roi)
    if not keep.any():
        raise ValueError("ROI has no visual support")
    blocked = torch.as_tensor(np.asarray(visual_positions)[~keep], dtype=torch.long)
    mask[:, :, start:end, blocked] = -torch.inf
    return int(keep.sum())


def move_inputs(inputs, device):
    return {key: value.to(device=device, dtype=torch.bfloat16)
            if value.is_floating_point() else value.to(device) for key, value in inputs.items()}


def collate_training(examples, pad_id):
    """Pad independent examples without cross-example attention or image-pooling offsets."""
    length = max(p["inputs"]["input_ids"].shape[1] for p in examples)
    batch = len(examples)
    ids = torch.full((batch, length), pad_id, dtype=torch.long)
    mask = torch.full((batch, 1, length, length), -torch.inf)
    positions, counts, targets, weights = [], [], [], []
    for index, packed in enumerate(examples):
        n = packed["inputs"]["input_ids"].shape[1]
        ids[index, :n] = packed["inputs"]["input_ids"][0]
        mask[index, :, :n, :n] = packed["inputs"]["attention_mask"][0]
        padding = torch.arange(n, length)
        mask[index, 0, padding, padding] = 0
        positions.extend((index, int(p)) for p in packed["positions"])
        counts.extend(packed["counts"].tolist())
        targets.extend(packed["targets"].tolist())
        weights.extend([1 / len(packed["targets"])] * len(packed["targets"]))
    inputs = {"input_ids": ids, "attention_mask": mask}
    for key in ("pixel_values", "image_token_pooling", "image_grids", "image_num_crops"):
        # Native pooling indices are local to each image; the model adds offsets itself.
        inputs[key] = torch.cat([p["inputs"][key] for p in examples], dim=0)
    return {"inputs": inputs, "positions": torch.tensor(positions), "counts": torch.tensor(counts),
            "targets": torch.tensor(targets), "loss_weights": torch.tensor(weights) / batch}


class SharedClassifier(nn.Module):
    def __init__(self, hidden_size, max_choices=32):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, max_choices)

    def forward(self, hidden, counts):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            logits = self.linear(self.norm(hidden.float()))
            invalid = torch.arange(logits.shape[-1], device=logits.device)[None] >= counts[:, None]
            return logits.masked_fill(invalid, -torch.inf)


class UnifiedSpatialModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone, self.classifier = backbone, classifier

    def forward(self, inputs, positions, counts):
        out = self.backbone(**inputs, use_cache=False)
        hidden = (out.last_hidden_state[positions[:, 0], positions[:, 1]] if positions.ndim == 2
                  else out.last_hidden_state[0, positions])
        return self.classifier(hidden, counts)


def build_unified_model(config, checkpoint=None, trainable=True):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText

    full = AutoModelForImageTextToText.from_pretrained(
        config["base_model"], trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    backbone, hidden = full.model, full.config.text_config.hidden_size
    del full
    if checkpoint:
        backbone = PeftModel.from_pretrained(backbone, str(Path(checkpoint) / "adapter"),
                                             is_trainable=trainable)
    else:
        backbone = get_peft_model(backbone, LoraConfig(
            r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"], target_modules=config["lora_targets"], bias="none"))
    classifier = SharedClassifier(hidden, config["max_choices"])
    if checkpoint:
        from safetensors.torch import load_file
        classifier.load_state_dict(load_file(str(Path(checkpoint) / "classifier.safetensors")))
    if trainable and config.get("gradient_checkpointing", True):
        backbone.get_base_model().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = UnifiedSpatialModel(backbone, classifier)
    if any(p.requires_grad for n, p in backbone.named_parameters() if "vision_backbone" in n):
        raise ValueError("Visual encoder/projector must be frozen in this ablation")
    if not trainable:
        model.requires_grad_(False).eval()
    return model


def save_unified_checkpoint(model, processor, config, output, step, optimizer=None):
    from safetensors.torch import save_file

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.backbone.save_pretrained(output / "adapter", safe_serialization=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.classifier.state_dict().items()},
              str(output / "classifier.safetensors"))
    processor.save_pretrained(output / "processor")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if optimizer:
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, output / "optimizer.pt")
    (output / "checkpoint.json").write_text(json.dumps({
        "step": step, "format": "base_plus_lora_plus_one_classifier",
        "point_variant": config["point_variant"], "base_revision": config["base_revision"],
        "calibrated": False}, indent=2) + "\n")


class UnifiedBuilder:
    def __init__(self, processor, config):
        self.processor, self.config = processor, config
        self.codebook = config["scalar_codebook"]
        self.variant = config["point_variant"]
        if self.variant not in ("single_image", "crop_refill", "roi_mask"):
            raise ValueError("Unknown point ablation")
        self.patch_id = processor.tokenizer.convert_tokens_to_ids("<im_patch>")
        self.header = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def render(self, text, image_count):
        content = [{"type": "image"} for _ in range(image_count)]
        content.append({"type": "text", "text": text})
        # Render each reveal separately. Rendering all messages at once hoists future crops.
        return self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)

    @staticmethod
    def load_images(inp):
        images = []
        for media in inp["media"]:
            with Image.open(media["uri"]) as image:
                images.append(image.convert("RGB"))
        return images

    @staticmethod
    def crop(original, box):
        w, h = original.size
        extent = (box[0] * w, box[1] * h, box[2] * w, box[3] * h)
        size = (max(1, round(extent[2] - extent[0])), max(1, round(extent[3] - extent[1])))
        return original.transform(size, Image.Transform.EXTENT, extent, Image.Resampling.BILINEAR)

    def point_prompt(self, inp, box, level):
        bounds = ", ".join(f"{x:.6f}" for x in box)
        options = "\n".join(f"{i+1}. {name}" for i, name in enumerate(POINT_NAMES))
        return (f"Request: {inp['question']}\nChoose one valid point satisfying the request. "
                f"Localization step {level+1} of 3. The current rectangle in the ORIGINAL image "
                f"is (left, top, right, bottom) = ({bounds}), normalized to [0, 1]. "
                "The origin is top-left, x increases rightward and y downward. "
                "Any image appended after the original is a zoomed view of its current rectangle. "
                "Divide the CURRENT rectangle into three equal rows and three equal columns. "
                f"Which region contains the point?\nOptions:\n{options}\nSelect exactly one option.")

    def scalar_prompt(self, inp, root=None):
        cb = codebook_for(inp, self.codebook)
        if cb.get("unit", "m") != "m":
            return numeric_prompt(inp, root, cb)
        if root is None:
            options = ["exactly zero meters"]
            options += [f"{lo:g} to {hi:g} meters" for lo, hi in zip(cb["edges_m"], cb["edges_m"][1:])]
            options[1] = f"greater than zero and less than {cb['edges_m'][1]:g} meters"
            options += [f"greater than {cb['edges_m'][-1]:g} meters"]
            instruction = "Select the range containing the nonnegative measurement."
        else:
            lo, hi = scalar_interval(root, cb)
            fine = scalar_fine_bins(root, cb)
            step = (hi - lo) / fine
            options = [f"{lo+i*step:.6g} to {lo+(i+1)*step:.6g} meters"
                       for i in range(fine)]
            instruction = f"Refine the previously selected range {lo:g} to {hi:g} meters."
        text = (f"Question: {inp['question']}\n{instruction} "
                "Ranges include the lower endpoint and exclude the upper endpoint, "
                "except that the maximum finite endpoint is included. Exact zero has its own class."
                "\nOptions:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(options))
                + "\nSelect exactly one option.")
        return text, len(options)

    def initial(self, inp, seed):
        task, mapping = inp["answer_space"]["kind"], []
        if task == "choice":
            order = list(range(len(inp["answer_space"]["options"])))
            if not self.config.get("preserve_option_order", False):
                random.Random(seed).shuffle(order)
            text, mapping = compile_prompt(inp, order)
            count = len(mapping)
        elif task == "scalar":
            text, count = self.scalar_prompt(inp)
        else:
            text, count = self.point_prompt(inp, ROOT_BOX, 0), 9
        return self.render(text, len(inp["media"])), count, mapping

    def encode(self, text, images, metadata=False, continuation=False):
        kwargs = {"text": [text], "return_tensors": "pt", "padding": False,
                  "return_mm_token_type_ids": True}
        if images:
            kwargs.update(images=images, return_pointing_metadata=metadata)
        encoded = self.processor(**kwargs)
        meta = encoded.pop("metadata", None)
        if continuation:
            bos = self.processor.tokenizer.bos_token_id or self.processor.tokenizer.eos_token_id
            if encoded["input_ids"][0, 0].item() != bos:
                raise ValueError("Unexpected continuation BOS")
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                encoded[key] = encoded[key][:, 1:]
        return dict(encoded), meta

    def geometry(self, inputs, metadata):
        positions = (inputs["input_ids"][0] == self.patch_id).nonzero().flatten().numpy()
        boxes = visual_token_boxes(metadata, inputs["image_grids"].tolist(),
                                   inputs["image_num_crops"].tolist(), inputs["pixel_values"].shape[1])
        if len(positions) != len(boxes):
            raise ValueError("Patch positions and geometric supports differ")
        return positions, boxes

    def training_example(self, row, seed):
        inp, task = row["input"], row["input"]["answer_space"]["kind"]
        text, count, mapping = self.initial(inp, seed)
        images = self.load_images(inp)
        original = images[0]
        if task == "choice":
            path = [mapping.index(row["target"]["option_id"])]
        elif task == "scalar":
            path = scalar_path(row["target"]["value"], self.codebook)
        else:
            path = point_path(row["target"]["points"][0])
        counts, boxes, box = [count], [ROOT_BOX], ROOT_BOX
        for stage in range(1, len(path)):
            if task == "point":
                box = child_box(box, path[stage - 1])
                question, count = self.point_prompt(inp, box, stage), 9
                extra = [self.crop(original, box)] if self.variant == "crop_refill" else []
            else:
                question, count = self.scalar_prompt(inp, path[0])
                extra = []
            text += f"{path[stage-1]+1}<|im_end|>\n" + self.render(question, len(extra))
            images.extend(extra)
            counts.append(count)
            boxes.append(box)
        needs_geometry = task == "point" and self.variant == "roi_mask"
        inputs, metadata = self.encode(text, images, metadata=needs_geometry)
        ids = inputs["input_ids"][0].tolist()
        positions = [i + len(self.header) - 1 for i in range(len(ids) - len(self.header) + 1)
                     if ids[i:i+len(self.header)] == self.header]
        if len(positions) != len(path):
            raise ValueError("Decision marker count mismatch")
        if len(ids) > self.config["max_sequence_length"]:
            raise ValueError("Sequence budget exceeded")
        types = inputs.pop("token_type_ids")[0]
        stages = torch.zeros(len(ids), dtype=torch.long)
        for stage in range(1, len(path)):
            stages[positions[stage-1]+1:positions[stage]+1] = stage
        mask = causal_visual_mask(len(ids), types, stages)
        if needs_geometry:
            visual_positions, supports = self.geometry(inputs, metadata)
            for stage in range(1, len(path)):
                apply_roi_mask(mask, visual_positions, supports, boxes[stage],
                               positions[stage-1]+1, positions[stage]+1)
        inputs["attention_mask"] = mask
        return {"inputs": inputs, "positions": torch.tensor(positions),
                "counts": torch.tensor(counts), "targets": torch.tensor(path),
                "mapping": mapping, "boxes": boxes}


@torch.inference_mode()
def predict_unified(model, builder, inp, device, seed, forced_path=None):
    """No targets are accepted. forced_path is solely for cache-equivalence integration tests."""
    task = inp["answer_space"]["kind"]
    images = builder.load_images(inp)
    original = images[0]
    text, count, mapping = builder.initial(inp, seed)
    need_geometry = task == "point" and builder.variant == "roi_mask"
    encoded, meta = builder.encode(text, images, metadata=need_geometry)
    geom = builder.geometry(encoded, meta) if need_geometry else None
    all_types, all_stages = [], []
    path, logits_trace, support_counts = [], [], []
    box, cache, previous_length, total_input_tokens = ROOT_BOX, None, 0, 0
    for stage in range(3 if task == "point" else 2 if task == "scalar" else 1):
        types = encoded.pop("token_type_ids")[0].tolist()
        all_types.extend(types)
        all_stages.extend([stage] * len(types))
        total_input_tokens += encoded["input_ids"].shape[1]
        if total_input_tokens > builder.config["max_sequence_length"]:
            raise ValueError("Rollout sequence budget exceeded")
        mask = causal_visual_mask(total_input_tokens, all_types, all_stages, previous_length)
        if need_geometry and stage > 0:
            support_counts.append(apply_roi_mask(mask, *geom, box))
        encoded["attention_mask"] = mask
        inputs = move_inputs(encoded, device)
        inputs["cache_position"] = torch.arange(previous_length, total_input_tokens, device=device)
        output = model.backbone(**inputs, past_key_values=cache, use_cache=True)
        logits = model.classifier(output.last_hidden_state[:, -1],
                                  torch.tensor([count], device=device))
        logits_trace.append(logits[0, :count].float().cpu().tolist())
        choice = int(logits.argmax(-1).item()) if forced_path is None else forced_path[stage]
        path.append(choice)
        cache, previous_length = output.past_key_values, total_input_tokens
        if task == "choice":
            break
        if task == "scalar":
            if stage == 1 or choice in (0, len(builder.codebook["edges_m"])):
                break
            question, count = builder.scalar_prompt(inp, choice)
            extra = []
        else:
            box = child_box(box, choice)
            if stage == 2:
                break
            question, count = builder.point_prompt(inp, box, stage + 1), 9
            extra = [builder.crop(original, box)] if builder.variant == "crop_refill" else []
        text = f"{choice+1}<|im_end|>\n" + builder.render(question, len(extra))
        encoded, _ = builder.encode(text, extra, continuation=True)
    if task == "choice":
        prediction = mapping[path[0]]
    elif task == "scalar":
        prediction = decode_scalar(path, builder.codebook)
    else:
        prediction = decode_point(path)
    return {"prediction": prediction, "path": path, "logits": logits_trace,
            "mapping": mapping, "input_tokens": total_input_tokens,
            "roi_support_counts": support_counts}
