"""Qwen3.5 multimodal backbone with the existing spatial decision contract.

Each QA is one causal row, with crops revealed at their actual decision stage.
This baseline deliberately uses complete-prefix rollout before cache optimization.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .hierarchy import ROOT_BOX, child_box, decode_point, decode_scalar, point_path, scalar_path
from .unified import SharedClassifier, UnifiedBuilder

LORA_TARGETS = (
    r"language_model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))"
)


def sample_seed(sample_id, seed):
    return seed ^ int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)


def train_mode(config):
    mode = config.get("train_mode", "lora")
    if mode not in ("lora", "full_language"):
        raise ValueError(f"Unsupported train mode: {mode}")
    return mode


def configure_full_language(backbone, trainable=True):
    """Keep the original vision/merger frozen; FP32 master weights for full updates."""
    backbone.requires_grad_(False)
    backbone.language_model.float().requires_grad_(trainable)


def make_processor(config):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        config["base_model"], local_files_only=config.get("local_files_only", False)
    )
    processor.tokenizer.padding_side = "right"
    processor.image_processor.size = {
        "shortest_edge": config["min_image_tokens"] * 32 * 32,
        "longest_edge": config["max_image_tokens"] * 32 * 32,
    }
    return processor


class Qwen35SpatialModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone, self.classifier = backbone, classifier

    def forward(self, inputs, positions, counts):
        out = self.backbone(**inputs, use_cache=False)
        hidden = out.last_hidden_state[positions[:, 0], positions[:, 1]]
        return self.classifier(hidden, counts)


def build_model(config, checkpoint=None, trainable=True):
    from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5TextModel

    mode = train_mode(config)
    full = Qwen3_5ForConditionalGeneration.from_pretrained(
        config["base_model"],
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    backbone, hidden = full.model, full.config.text_config.hidden_size
    del full
    backbone.requires_grad_(False)
    if mode == "full_language":
        if checkpoint:
            text_directory = Path(checkpoint) / "language_model"
            text_config = Qwen3_5TextConfig.from_pretrained(text_directory, local_files_only=True)
            backbone.language_model = Qwen3_5TextModel.from_pretrained(
                text_directory,
                config=text_config,
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
            backbone.config.text_config = backbone.language_model.config
        configure_full_language(backbone, trainable)
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

    classifier = seed_head(config, lambda: SharedClassifier(hidden, config["max_choices"]))
    if checkpoint:
        from safetensors.torch import load_file

        classifier.load_state_dict(load_file(str(Path(checkpoint) / "classifier.safetensors")))
    if trainable and config.get("gradient_checkpointing", False):
        native = backbone.get_base_model() if mode == "lora" else backbone
        native.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = Qwen35SpatialModel(backbone, classifier)
    if any(
        p.requires_grad
        for n, p in backbone.named_parameters()
        if n.startswith("visual.") or ".visual." in n
    ):
        raise ValueError("Vision encoder and merger must remain frozen")
    if trainable and not any(
        "linear_attn" in n and p.requires_grad for n, p in backbone.named_parameters()
    ):
        raise ValueError("DeltaNet must have trainable parameters")
    if not trainable:
        model.requires_grad_(False).eval()
    return model


def move_inputs(inputs, device):
    return {
        key: value.to(
            device,
            dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
            non_blocking=True,
        )
        for key, value in inputs.items()
    }


class Qwen35Builder(UnifiedBuilder):
    def __init__(self, processor, config):
        self.processor, self.config = processor, config
        self.codebook, self.variant = config["scalar_codebook"], "crop_refill"
        rendered = self.render("Example question", 0)
        suffix = rendered[rendered.rindex("<|im_start|>assistant") :]
        self.header = processor.tokenizer.encode(suffix, add_special_tokens=False)

    def render(self, text, image_count):
        content = [{"type": "image"} for _ in range(image_count)]
        content.append({"type": "text", "text": text})
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def training_spec(self, row, seed):
        inp, task = row["input"], row["input"]["answer_space"]["kind"]
        text, count, mapping = self.initial(inp, seed)
        images = self.load_images(inp)
        original, box = images[0], ROOT_BOX
        if task == "choice":
            path = [mapping.index(row["target"]["option_id"])]
        elif task == "scalar":
            path = scalar_path(row["target"]["value"], self.codebook)
        else:
            path = point_path(row["target"]["points"][0])
        counts = [count]
        for stage in range(1, len(path)):
            if task == "point":
                box = child_box(box, path[stage - 1])
                question, count = self.point_prompt(inp, box, stage), 9
                extra = [self.crop(original, box)]
            else:
                question, count = self.scalar_prompt(inp, path[0])
                extra = []
            text += f"{path[stage - 1] + 1}<|im_end|>\n" + self.render(question, len(extra))
            images.extend(extra)
            counts.append(count)
        return {
            "text": text,
            "images": images,
            "counts": counts,
            "targets": path,
            "sample_id": row["sample_id"],
            "weight": row.get("_weight", 1.0),
        }

    def encode_specs(self, specs, training=True):
        encoded = dict(
            self.processor(
                text=[s["text"] for s in specs],
                images=[im for s in specs for im in s["images"]],
                padding=True,
                return_tensors="pt",
                device="cpu",
            )
        )
        encoded.pop("token_type_ids", None)
        lengths = encoded["attention_mask"].sum(-1).tolist()
        if max(lengths) > self.config["max_sequence_length"]:
            raise ValueError(f"Sequence budget exceeded without truncation: {lengths}")
        positions, counts, targets, weights = [], [], [], []
        if training:
            for index, spec in enumerate(specs):
                ids = encoded["input_ids"][index, : lengths[index]].tolist()
                ends = [
                    i + len(self.header) - 1
                    for i in range(len(ids) - len(self.header) + 1)
                    if ids[i : i + len(self.header)] == self.header
                ]
                if len(ends) != len(spec["targets"]):
                    raise ValueError(f"Decision positions mismatch: {spec['sample_id']}")
                positions.extend((index, p) for p in ends)
                counts.extend(spec["counts"])
                targets.extend(spec["targets"])
                weights.extend([spec["weight"] / len(ends)] * len(ends))
        else:
            positions = [(i, n - 1) for i, n in enumerate(lengths)]
            counts = [s["count"] for s in specs]
        return {
            "inputs": encoded,
            "positions": torch.tensor(positions),
            "counts": torch.tensor(counts),
            "targets": torch.tensor(targets),
            "weights": torch.tensor(weights, dtype=torch.float32),
            "qa_count": sum(s.get("weight", 1.0) for s in specs),
            "tokens": sum(lengths),
            "padded_tokens": encoded["input_ids"].numel(),
        }

    def collate(self, rows):
        return self.encode_specs(
            [
                self.training_spec(row, sample_seed(row["sample_id"], self.config["seed"]))
                for row in rows
            ]
        )


@torch.inference_mode()
def predict_batch(model, builder, requests, device):
    """Free rollout: this function accepts inputs and IDs, never target labels."""
    states, results = [], [None] * len(requests)
    for index, request in enumerate(requests):
        inp = request["input"]
        text, count, mapping = builder.initial(
            inp, sample_seed(request["id"], builder.config["seed"])
        )
        states.append(
            {
                "index": index,
                "input": inp,
                "text": text,
                "count": count,
                "mapping": mapping,
                "images": builder.load_images(inp),
                "path": [],
                "logits": [],
                "box": ROOT_BOX,
            }
        )
    while states:
        packed = builder.encode_specs(states, training=False)
        logits = model(
            move_inputs(packed["inputs"], device),
            packed["positions"].to(device),
            packed["counts"].to(device),
        )
        choices = logits.argmax(-1).tolist()
        following = []
        for state, choice, scores in zip(states, choices, logits.float().cpu()):
            path, inp = state["path"], state["input"]
            path.append(choice)
            state["logits"].append(scores[: state["count"]].tolist())
            task = inp["answer_space"]["kind"]
            done = (
                task == "choice"
                or (task == "point" and len(path) == 3)
                or (
                    task == "scalar"
                    and (len(path) == 2 or choice in (0, len(builder.codebook["edges_m"])))
                )
            )
            if done:
                prediction = (
                    state["mapping"][choice]
                    if task == "choice"
                    else decode_point(path)
                    if task == "point"
                    else decode_scalar(path, builder.codebook)
                )
                results[state["index"]] = {
                    "prediction": prediction,
                    "path": path,
                    "logits": state["logits"],
                }
                continue
            if task == "point":
                state["box"] = child_box(state["box"], choice)
                question, count = builder.point_prompt(inp, state["box"], len(path)), 9
                extra = [builder.crop(state["images"][0], state["box"])]
            else:
                question, count = builder.scalar_prompt(inp, choice)
                extra = []
            state["text"] += f"{choice + 1}<|im_end|>\n" + builder.render(question, len(extra))
            state["images"].extend(extra)
            state["count"] = count
            following.append(state)
        states = following
    return results


def save_checkpoint(model, processor, config, path, step):
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    mode = train_mode(config)
    if mode == "full_language":
        model.backbone.language_model.save_pretrained(
            path / "language_model", safe_serialization=True, max_shard_size="4GB"
        )
    else:
        model.backbone.save_pretrained(path / "adapter", safe_serialization=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.classifier.state_dict().items()},
        str(path / "classifier.safetensors"),
    )
    processor.save_pretrained(path / "processor")
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (path / "checkpoint.json").write_text(
        json.dumps(
            {
                "step": step,
                "format": f"qwen35-{mode}-shared-classifier",
                "train_mode": mode,
                "base_revision": config["base_revision"],
            },
            indent=2,
        )
        + "\n"
    )
