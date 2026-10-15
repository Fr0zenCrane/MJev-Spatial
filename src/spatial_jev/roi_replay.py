"""Local original-image token replay, optionally transplanting attention K/V.

No pixels are cropped/resized after the original image. Copy merged vision
tokens whose support intersects the current rectangle, in original row-major
order, retaining original M-RoPE coordinates. The KV variant also gathers the
original post-RoPE K and V at every intact full-attention layer. DeltaNet, Q,
gates, residuals and MLPs are computed normally; this is a representation ablation,
not a claim that all computations at replayed token slots have been eliminated.
"""

import math

import torch

VIT_REPLAY = 'roi_vit_replay'
KV_REPLAY = 'roi_attention_kv_replay'
REPLAY_VARIANTS = (VIT_REPLAY, KV_REPLAY)
EMPTY_ROI_IMAGE = '<|vision_start|><|vision_end|>'


def roi_cells(box, shape):
    """Positive-area overlap with a normalized rectangle, no target-dependent zoom."""
    h, w = map(int, shape)
    x0, y0, x1, y1 = box
    if h < 1 or w < 1 or not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('Invalid original-image ROI/grid')
    # Avoid including a neighboring cell from floating-point noise at exact edges.
    left, top = math.floor(x0*w+1e-9), math.floor(y0*h+1e-9)
    right, bottom = math.ceil(x1*w-1e-9), math.ceil(y1*h-1e-9)
    return [y*w+x for y in range(top, bottom) for x in range(left, right)]


def expand_replay_inputs(builder, specs, encoded):
    """Insert virtual local-image slots after processing only the real original(s).

Base input tensors are retained solely for authoritative native M-RoPE creation;
gather maps align expanded slots, original visual features and original positions.
The ordinary causal mask still uses SEQUENCE order, not repeated spatial IDs.
"""
    if not any(s.get('roi_boxes') for s in specs):
        return encoded
    tokenizer = builder.processor.tokenizer
    image_id, start_id, end_id = [tokenizer.convert_tokens_to_ids(s) for s in
                                ('<|image_pad|>', '<|vision_start|>', '<|vision_end|>')]
    merge = builder.processor.image_processor.merge_size
    original_ids, original_mask = encoded['input_ids'], encoded['attention_mask']
    original_types = encoded.get('mm_token_type_ids', (original_ids == image_id).long())
    new_ids, rope_sources, kv_sources, feature_sources = [], [], [], []
    grid_cursor = feature_cursor = 0
    for row, spec in enumerate(specs):
        n = int(original_mask[row].sum())
        ids = original_ids[row, :n].tolist()
        grids = encoded['image_grid_thw'][grid_cursor:grid_cursor+len(spec['images'])]
        n_features = sum(int(g.prod())//merge**2 for g in grids)
        image_positions = [i for i, token in enumerate(ids) if token == image_id]
        if len(image_positions) != n_features:
            raise ValueError('Original visual token/grid count mismatch')
        original_feature = {pos: feature_cursor+i for i, pos in enumerate(image_positions)}
        boxes = spec.get('roi_boxes', [])
        selected = []
        if boxes:
            if len(grids) != 1 or int(grids[0, 0]) != 1:
                raise ValueError('ROI replay requires one still original image')
            shape = (int(grids[0, 1])//merge, int(grids[0, 2])//merge)
            selected = [roi_cells(box, shape) for box in boxes]
        tokens, positions, aliases = [], [], []
        slot = 0
        for i, token in enumerate(ids):
            if token == end_id and i and ids[i-1] == start_id:
                if slot >= len(selected):
                    raise ValueError('Unregistered local-image slot')
                for cell in selected[slot]:
                    source = image_positions[cell]
                    tokens.append(image_id)
                    positions.append(source)
                    # All originals precede every insertion, hence their sequence
                    # indices remain identical in the expanded row.
                    if source >= len(tokens)-1:
                        raise ValueError('Replay may only reference earlier original tokens')
                    aliases.append(source)
                    feature_sources.append(feature_cursor+cell)
                slot += 1
            tokens.append(token)
            positions.append(i)
            aliases.append(len(tokens)-1)
            if token == image_id:
                feature_sources.append(original_feature[i])
        if slot != len(boxes):
            raise ValueError('Missing local-image marker')
        new_ids.append(tokens)
        rope_sources.append(positions)
        kv_sources.append(aliases)
        grid_cursor += len(grids)
        feature_cursor += n_features
    length = max(map(len, new_ids))
    if length > builder.config['max_sequence_length']:
        raise ValueError('ROI replay exceeds sequence budget; refusing truncation')
    batch = len(specs)
    ids = original_ids.new_full((batch, length), tokenizer.pad_token_id)
    mask = original_mask.new_zeros((batch, length))
    rope = original_ids.new_zeros((batch, length))
    aliases = torch.arange(length, dtype=original_ids.dtype).expand(batch, -1).clone()
    for row, tokens in enumerate(new_ids):
        n = len(tokens)
        ids[row, :n] = torch.tensor(tokens)
        mask[row, :n] = 1
        rope[row, :n] = torch.tensor(rope_sources[row])
        aliases[row, :n] = torch.tensor(kv_sources[row])
    encoded.update(input_ids=ids, attention_mask=mask, mm_token_type_ids=(ids == image_id).long(),
                   roi_base_ids=original_ids, roi_base_mask=original_mask, roi_base_types=original_types,
                   roi_position_sources=rope, roi_kv_sources=aliases,
                   roi_feature_sources=torch.tensor(feature_sources, dtype=torch.long))
    return encoded


class CopyAttentionKV:
    """Stateless, differentiable aliasing at the native post-RoPE cache boundary."""

    def __init__(self, sources):
        self.sources = sources
        self.calls = 0

    def update(self, keys, values, layer_idx, *args, **kwargs):
        if keys.shape[:3] != values.shape[:3] or keys.shape[0] != self.sources.shape[0] or keys.shape[2] != self.sources.shape[1]:
            raise ValueError('KV replay shape mismatch')
        self.calls += 1
        def gather(tensor):
            indices = self.sources[:, None, :, None].expand(-1, tensor.shape[1], -1, tensor.shape[-1])
            return tensor.gather(2, indices)
        return gather(keys), gather(values)


def kv_replay_hidden(lm, hidden, positions, attention_mask, sources):
    """Scoped native-attention hooks; never modify recurrent/conv states or weights."""
    if lm.gradient_checkpointing:
        raise ValueError('Scoped KV replay requires gradient_checkpointing=False')
    cache, handles, count = CopyAttentionKV(sources), [], 0
    def inject(module, args, kwargs):
        if kwargs.get('past_key_values') is not None:
            raise ValueError('ROI replay must not mix with a persistent prefix cache')
        return args, {**kwargs, 'past_key_values': cache}
    try:
        for layer in lm.layers:
            if layer.block_type == 'full_attention':
                count += 1
                handles.append(layer.self_attn.register_forward_pre_hook(inject, with_kwargs=True))
        result = lm(inputs_embeds=hidden, position_ids=positions,
                    attention_mask=attention_mask, use_cache=False).last_hidden_state
        if cache.calls != count or not count:
            raise ValueError('Every intact full-attention layer must copy its own K/V')
        return result
    finally:
        for handle in handles:
            handle.remove()
