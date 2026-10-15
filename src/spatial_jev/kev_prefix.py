"""Functional hybrid prefixes with native multimodal embeddings and 3-axis RoPE.

Prefix and the layer-wise training schedule are adapted from
jaredpalmer/kev@f1535963cea021439370c23127bc970b6788e730/kev/shared_prefix.py
(Apache-2.0; see third_party/kev/). Branches never mutate a saved prefix.
"""

import os
from collections import OrderedDict

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Prefix:
    record_past = True

    def __init__(self):
        self.keys = self.values = self.conv = None
        self.recurrent_states = {}

    @property
    def layers(self):
        return self

    def __getitem__(self, layer_idx):
        return self

    def has_previous_state(self, layer_idx=None, state_idx=0):
        return self.conv is not None

    def update_conv_state(self, conv_input, layer_idx, conv_kernel_size, **kwargs):
        if self.conv is not None:
            return torch.cat([self.conv, conv_input], -1)
        self.conv = F.pad(conv_input, (max(0, conv_kernel_size - conv_input.shape[-1]), 0))[
            ..., -conv_kernel_size:
        ]
        return conv_input

    def update_recurrent_state(self, recurrent_state, layer_idx, **kwargs):
        self.recurrent_states.setdefault(0, recurrent_state)
        return recurrent_state

    def update(self, keys, values, layer_idx, *args, **kwargs):
        if self.keys is None:
            self.keys, self.values = keys, values
            return keys, values
        return torch.cat([self.keys, keys], -2), torch.cat([self.values, values], -2)

    def branch(self, owner):
        other = Prefix()
        pick = lambda t: None if t is None else t.index_select(0, owner)
        other.keys, other.values, other.conv = pick(self.keys), pick(self.values), pick(self.conv)
        other.recurrent_states = {k: pick(v) for k, v in self.recurrent_states.items()}
        return other

    def nbytes(self):
        tensors = [self.keys, self.values, self.conv, *self.recurrent_states.values()]
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    def compact(self):
        # Cached slices must own their storage so byte accounting does not hide a
        # complete prefill tensor behind a small convolution window or KV view.
        self.keys = None if self.keys is None else self.keys.clone().contiguous()
        self.values = None if self.values is None else self.values.clone().contiguous()
        self.conv = None if self.conv is None else self.conv.clone().contiguous()
        self.recurrent_states = {
            k: v.clone().contiguous() for k, v in self.recurrent_states.items()
        }
        return self


def attention_mask(allow, hidden, implementation):
    allow = allow[:, None]
    if implementation == "sdpa":
        return allow
    return hidden.new_zeros(allow.shape).masked_fill(~allow, torch.finfo(hidden.dtype).min)


def split_rows(hidden, positions, packed):
    prefixes = packed["prefix_lengths"].tolist()
    lengths = packed["lengths"].tolist()
    unique, first, owners = {}, [], []
    for i, key in enumerate(packed["state_keys"]):
        if key not in unique:
            unique[key] = len(first)
            first.append(i)
        reference = first[unique[key]]
        n = prefixes[i]
        if n != prefixes[reference] or not torch.equal(
            packed["inputs"]["input_ids"][i, :n], packed["inputs"]["input_ids"][reference, :n]
        ):
            raise ValueError("Same state identity compiled to different prefix tokens")
        if not torch.equal(positions[:, i, :n], positions[:, reference, :n]):
            raise ValueError("Same state identity compiled to different multimodal positions")
        owners.append(unique[key])
    ns, nb = max(prefixes), max(n - p for n, p in zip(lengths, prefixes))
    s = hidden.new_zeros((len(first), ns, hidden.shape[-1]))
    b = hidden.new_zeros((len(lengths), nb, hidden.shape[-1]))
    sp = positions.new_zeros((3, len(first), ns))
    bp = positions.new_zeros((3, len(lengths), nb))
    sm = torch.zeros(s.shape[:2], dtype=torch.bool, device=hidden.device)
    bm = torch.zeros(b.shape[:2], dtype=torch.bool, device=hidden.device)
    for j, i in enumerate(first):
        n = prefixes[i]
        s[j, -n:], sp[:, j, -n:], sm[j, -n:] = hidden[i, :n], positions[:, i, :n], True
    for i, (p, n) in enumerate(zip(prefixes, lengths)):
        b[i, : n - p], bp[:, i, : n - p], bm[i, : n - p] = (
            hidden[i, p:n],
            positions[:, i, p:n],
            True,
        )
    return s, b, sp, bp, sm, bm, torch.tensor(owners, device=hidden.device)


def masks(lm, s, b, sm, bm, owner):
    ns, nb, device = s.shape[1], b.shape[1], s.device
    causal = lambda n: torch.ones(n, n, dtype=torch.bool, device=device).tril()
    eye = lambda n: torch.eye(n, dtype=torch.bool, device=device)
    impl = lm.config._attn_implementation
    s_allow = (causal(ns)[None] & sm[:, None, :]) | eye(ns)[None]
    b_allow = torch.cat(
        [
            sm[owner][:, None, :].expand(-1, nb, -1),
            (causal(nb)[None] & bm[:, None, :]) | eye(nb)[None],
        ],
        -1,
    )
    ms = {
        "full_attention": None
        if impl == "sdpa" and bool(sm.all())
        else attention_mask(s_allow, s, impl),
        "linear_attention": sm.to(s.dtype),
    }
    mb = {"full_attention": attention_mask(b_allow, b, impl), "linear_attention": bm.to(b.dtype)}
    return ms, mb


def shared_hidden(lm, hidden, positions, packed):
    s, b, sp, bp, sm, bm, owner = split_rows(hidden, positions, packed)
    ms, mb = masks(lm, s, b, sm, bm, owner)
    rs, rb = lm.rotary_emb(s, sp), lm.rotary_emb(b, bp)
    for layer in lm.layers:
        # Capture the current layer, never a late-bound loop variable in recomputation.
        def step(s, b, layer=layer):
            prefix = Prefix()
            s = layer.forward(
                s,
                position_embeddings=rs,
                attention_mask=ms[layer.block_type],
                past_key_values=prefix,
            )
            b = layer.forward(
                b,
                position_embeddings=rb,
                attention_mask=mb[layer.block_type],
                past_key_values=prefix.branch(owner),
            )
            return s * sm[..., None], b

        if lm.gradient_checkpointing and lm.training and torch.is_grad_enabled():
            s, b = checkpoint(step, s, b, use_reentrant=False)
        else:
            s, b = step(s, b)
    return lm.norm(b)


class ByteLRU:
    def __init__(self, max_bytes):
        self.max_bytes, self.bytes = max_bytes, 0
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def put(self, key, value, size):
        if size > self.max_bytes:
            return
        if key in self.entries:
            self.bytes -= self.entries.pop(key)[1]
        while self.entries and self.bytes + size > self.max_bytes:
            self.bytes -= self.entries.popitem(last=False)[1][1]
        self.entries[key] = (value, size)
        self.bytes += size

    def clear(self):
        self.entries.clear()
        self.bytes = 0


class KevRuntime:
    """An immutable-model session with byte-bounded image and hybrid prefix caches.

    Every round recomputes only the private question history from the original state.
    All branches of one state are batched; attention/conv/recurrent prefixes are read-only.
    """

    def __init__(self, model, config, prefix_bytes=8 * 1024**3, image_bytes=2 * 1024**3,
                 enable_native_qk=False):
        from .roi_replay import REPLAY_VARIANTS
        if config.get('point_variant') in REPLAY_VARIANTS:
            raise ValueError('ROI replay uses uncached full rows, not the persistent hybrid-prefix runtime')
        if config.get("point_variant") == "visual_tokens":
            raise ValueError("Visual-token pointing requires prefix keys; cached branch-only readout is unsupported")
        if not enable_native_qk and config.get("readout_kind") in (
            "native_last_qk_mean",
            "native_merged_qk256",
            "native_last_qk_concat",
        ):
            raise ValueError("Native QK currently uses the uncached causal-row forward")
        if enable_native_qk and (config.get("readout_kind") != "native_last_qk_mean"
                or config.get("readout_layer_indices") is not None
                or config.get("gradient_checkpointing", False)):
            raise ValueError("Experimental native cache requires single-layer mean and no checkpointing")
        import hashlib
        import json

        self.model = model
        self.signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.versions = tuple((p._version, p.dtype, p.device) for p in model.parameters())
        self.prefix_cache, self.image_cache = ByteLRU(prefix_bytes), ByteLRU(image_bytes)

    def clear(self):
        self.prefix_cache.clear()
        self.image_cache.clear()

    @torch.inference_mode()
    def __call__(self, packed):
        if packed.get("visual_pointing", False):
            raise ValueError("Visual-token pointing requires the uncached causal-row forward")
        if self.model.training or self.versions != tuple(
            (p._version, p.dtype, p.device) for p in self.model.parameters()
        ):
            raise ValueError("Create a fresh runtime after changing model weights or training mode")
        model, inputs = self.model, packed["inputs"]
        device_type = inputs["input_ids"].device.type
        precision = (
            torch.is_autocast_enabled(device_type),
            str(torch.get_autocast_dtype(device_type)),
            torch.get_float32_matmul_precision(),
            os.environ.get("TRITON_F32_DEFAULT", "tf32"),
        )
        scope = (self.signature, precision)
        features = [self.image_cache.get((scope, k)) for k in packed["image_keys"]]
        missing = [i for i, f in enumerate(features) if f is None]
        if missing:
            chunks = inputs["pixel_values"].split(inputs["unique_grid_thw"].prod(-1).tolist())
            fs = model.native.get_image_features(
                torch.cat([chunks[i] for i in missing]),
                inputs["unique_grid_thw"][missing],
                return_dict=True,
            ).pooler_output
            for i, f in zip(missing, fs):
                f = f.clone().contiguous()
                features[i] = f
                self.image_cache.put(
                    (scope, packed["image_keys"][i]), f, f.numel() * f.element_size()
                )
        hidden, positions = model.embeddings(inputs, features=features)
        with model.language_context(hidden.device.type):
            return self.language_forward(packed, hidden, positions, scope)

    def language_forward(self, packed, hidden, positions, scope):
        model, inputs = self.model, packed["inputs"]
        s, b, sp, bp, sm, bm, owner = split_rows(hidden, positions, packed)
        lm = model.native.language_model
        result = torch.zeros_like(b)
        for state in range(len(s)):
            rows = (owner == state).nonzero().flatten()
            first = int(rows[0])
            ns = int(packed["prefix_lengths"][first])
            key = (
                scope,
                packed["state_keys"][first],
                str(hidden.dtype),
                tuple(inputs["input_ids"][first, :ns].tolist()),
                tuple(positions[:, first, :ns].flatten().tolist()),
            )
            entry = self.prefix_cache.get(key)
            if entry is None:
                hs = s[state : state + 1, -ns:]
                rs = lm.rotary_emb(hs, sp[:, state : state + 1, -ns:])
                entry = []
                for layer in lm.layers:
                    prefix = Prefix()
                    hs = layer.forward(hs, position_embeddings=rs, past_key_values=prefix)
                    entry.append(prefix.compact())
                self.prefix_cache.put(key, entry, sum(p.nbytes() for p in entry))
            # Different states need not inherit the longest unrelated question's
            # padding. Their final readouts only index these real branch lengths.
            branch_length = int(bm[rows].sum(-1).max())
            hb, mask = b[rows, :branch_length], bm[rows, :branch_length]
            rb = lm.rotary_emb(hb, bp[:, rows, :branch_length])
            owners = torch.zeros(len(rows), dtype=torch.long, device=hb.device)
            _, mb = masks(
                lm, s[state : state + 1, -ns:], hb, sm[state : state + 1, -ns:], mask, owners
            )
            for layer, prefix in zip(lm.layers, entry):
                hb = layer.forward(
                    hb,
                    position_embeddings=rb,
                    attention_mask=mb[layer.block_type],
                    past_key_values=prefix.branch(owners),
                )
            result[rows, :branch_length] = lm.norm(hb)
        if model.pointer is None:
            # The cached image prefix is unchanged. Candidates and decisions are
            # private text suffix tokens; retain their ORIGINAL M-RoPE coordinates
            # while translating tensor indices into the returned suffix buffer.
            selected = dict(packed)
            for key in ("positions", "option_positions"):
                indices = packed[key].clone()
                indices[:, 1] -= packed["prefix_lengths"][indices[:, 0]]
                if (indices[:, 1] < 0).any():
                    raise ValueError("Native suffix cache cannot read keys inside the prefix")
                selected[key] = indices
            return lm.native_qk(result, selected, lm.rotary_emb(result, bp))
        return model.readout(result, packed, branch_only=True)
