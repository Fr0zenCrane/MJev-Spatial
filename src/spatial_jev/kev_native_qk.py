"""Spatial decisions from the pretrained final attention's native Q/K scores.

Only existing parameters survive: the first N-1 layers and the last input norm,
Q-only rows, K projection and Q/K norms. No final model norm or learned pointer.
"""

from contextlib import nullcontext

import torch
from torch import nn
from transformers import Qwen3_5TextModel
from transformers.models.qwen3_5.modeling_qwen3_5 import rotate_half

NATIVE_QK = "native_last_qk_mean"
MERGED_QK = "native_merged_qk256"
CONCAT_QK = "native_last_qk_concat"
READOUT_KINDS = (NATIVE_QK, MERGED_QK, CONCAT_QK)


class NativeQKReadout(nn.Module):
    def __init__(self, layer, merge_heads=False, concat_heads=False):
        super().__init__()
        if merge_heads and concat_heads:
            raise ValueError("Weight merging and feature concatenation are distinct readouts")
        self.concat_heads = concat_heads
        if layer.block_type != "full_attention":
            raise ValueError("Native QK readout requires a final full-attention layer")
        attention = layer.self_attn
        if attention.q_proj.bias is not None or attention.k_proj.bias is not None:
            raise ValueError("This readout expects the Qwen3.5 bias-free projections")
        self.head_dim = attention.head_dim
        self.groups = attention.num_key_value_groups
        self.heads = attention.q_proj.out_features // (2 * self.head_dim)
        self.kv_heads = self.heads // self.groups
        self.scale = attention.scaling
        self.readout_fp32 = False
        self.input_layernorm = layer.input_layernorm
        self.k_proj = attention.k_proj
        self.q_norm, self.k_norm = attention.q_norm, attention.k_norm
        # Q and output gate are interleaved PER HEAD, not two global halves.
        weight = attention.q_proj.weight
        selected = weight.view(self.heads, 2 * self.head_dim, -1)[:, : self.head_dim]
        if merge_heads:
            # Average PRETRAINED WEIGHTS in FP32, before applying the shared
            # Q/K Norms. Averaging the four GQA keys equals averaging their
            # sixteen aligned copies. No learned fusion or random initialization.
            selected = selected.float().mean(dim=0)
            key_weight = attention.k_proj.weight
            merged_key = key_weight.view(self.kv_heads, self.head_dim, -1).float().mean(dim=0)
            self.k_proj = nn.Linear(weight.shape[1], self.head_dim, bias=False, device="meta")
            self.k_proj.weight = nn.Parameter(
                merged_key.detach().clone(), requires_grad=key_weight.requires_grad
            )
            self.heads = self.kv_heads = self.groups = 1
        self.q_proj = nn.Linear(
            weight.shape[1], self.heads * self.head_dim, bias=False, device="meta"
        )
        self.q_proj.weight = nn.Parameter(
            selected.reshape(self.heads * self.head_dim, -1).detach().clone(),
            requires_grad=weight.requires_grad,
        )

    def readout_parameters(self):
        """Merged projections and their norms; input_layernorm stays in backbone."""
        for module in (self.q_proj, self.k_proj, self.q_norm, self.k_norm):
            yield from module.parameters()

    @staticmethod
    def rotate(states, indices, cos, sin):
        # Same partial RoPE as native apply_rotary_pos_emb; gather original
        # multimodal positions rather than renumbering selected candidates.
        cos = cos[indices[:, 0], indices[:, 1]][:, None]
        sin = sin[indices[:, 0], indices[:, 1]][:, None]
        width = cos.shape[-1]
        rotated, unchanged = states[..., :width], states[..., width:]
        return torch.cat((rotated * cos + rotate_half(rotated) * sin, unchanged), dim=-1)

    def forward(self, hidden, packed, position_embeddings):
        context = (
            torch.autocast(device_type=hidden.device.type, enabled=False)
            if self.readout_fp32
            else nullcontext()
        )
        with context:
            return self._scores(hidden, packed, position_embeddings)

    def _scores(self, hidden, packed, position_embeddings):
        p, o = packed["positions"], packed["option_positions"]
        decisions, options = hidden[p[:, 0], p[:, 1]], hidden[o[:, 0], o[:, 1]]
        if self.readout_fp32:
            decisions, options = decisions.float(), options.float()
            position_embeddings = tuple(p.float() for p in position_embeddings)
        decisions = self.input_layernorm(decisions)
        options = self.input_layernorm(options)
        queries = self.q_norm(self.q_proj(decisions).view(-1, self.heads, self.head_dim))
        keys = self.k_norm(self.k_proj(options).view(-1, self.kv_heads, self.head_dim))
        queries = self.rotate(queries, p, *position_embeddings)
        keys = self.rotate(keys, o, *position_embeddings)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            keys = keys.float().repeat_interleave(self.groups, dim=1)
            if self.concat_heads:
                # Retain all per-head Norm/RoPE operations, then concatenate
                # channels on BOTH sides. Repeated GQA keys stay weight-tied.
                # For H heads this is sqrt(H) times the mean-head logit,
                # not cross-head mixing. No additional norm or learned scale.
                query_features = queries.float().flatten(start_dim=1)
                key_features = keys.flatten(start_dim=1)
                scores = (query_features[packed["owner"]] * key_features).sum(-1)
                scores = scores * (self.heads * self.head_dim) ** -0.5
            else:
                scores = (queries.float()[packed["owner"]] * keys).sum(-1) * self.scale
                # Average logits, NOT per-head probabilities. Candidate masking
                # is independent for every QA/stage; padded slots carry no mass.
                scores = scores.mean(dim=-1)
            logits = scores.new_full((len(p), int(packed["counts"].max())), -torch.inf)
            return logits.index_put((packed["owner"], packed["slot"]), scores)


def prune_native_qk(language_model, kind=NATIVE_QK, layer_index=None):
    if kind not in READOUT_KINDS:
        raise ValueError(f"Unsupported native readout: {kind}")
    if hasattr(language_model, "native_qk"):
        raise ValueError("Native QK pruning was already applied")
    layer_index = len(language_model.layers) - 1 if layer_index is None else layer_index
    if not isinstance(layer_index, int) or not 0 <= layer_index < len(language_model.layers):
        raise ValueError(f"Invalid readout layer index: {layer_index}")
    last = language_model.layers[layer_index]
    language_model.native_qk = NativeQKReadout(
        last, merge_heads=kind == MERGED_QK, concat_heads=kind == CONCAT_QK
    )
    language_model.native_qk.readout_fp32 = getattr(
        language_model.config, "kev_readout_fp32", False
    )
    language_model.layers = nn.ModuleList(list(language_model.layers[:layer_index]))
    # Execute only the original layers BEFORE the selected full attention.
    # Later layers cannot affect this readout and must not enter the optimizer.
    language_model.norm = nn.Identity()
    # Keep the original layer configuration for checkpoint reconstruction.
    language_model.config.kev_readout_kind = kind
    language_model.config.kev_readout_layer_index = layer_index
    return language_model


class NativeQKTextModel(Qwen3_5TextModel):
    """Explicit loader for the pruned checkpoint; never load it as an ordinary LM."""

    def __init__(self, config):
        if getattr(config, "kev_readout_kind", None) not in READOUT_KINDS:
            raise ValueError("Missing native QK checkpoint architecture marker")
        super().__init__(config)
        prune_native_qk(
            self,
            config.kev_readout_kind,
            getattr(config, "kev_readout_layer_index", config.num_hidden_layers - 1),
        )
