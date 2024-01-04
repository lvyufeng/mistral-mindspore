import gc
import mindspore
from mindspore import ops
from typing import List, Tuple
from dataclasses import dataclass

from mistral.attn_bias import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)

@dataclass
class RotatingCacheInputMetadata:
    # rope absolute positions
    positions: mindspore.Tensor
    # which elements in the sequences need to be cached
    to_cache_mask: mindspore.Tensor
    # how many elements are cached per sequence
    cached_elements: mindspore.Tensor
    # where tokens should go in the cache
    cache_positions: mindspore.Tensor

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]

def replace_references(old_obj, new_obj):
    """use replace_references instead of Tensor.set_data due to mindspore errors."""
    # Get all objects referring to old_obj
    referrers = gc.get_referrers(old_obj)

    # Replace references
    for referrer in referrers:
        if isinstance(referrer, dict):
            # If the reference is in a dictionary
            for key, value in referrer.items():
                if value is old_obj:
                    referrer[key] = new_obj
        elif isinstance(referrer, list):
            # If the reference is in a list or tuple
            index = referrer.index(old_obj)
            referrer[index] = new_obj
        elif isinstance(referrer, tuple):
            pass
        elif hasattr(referrer, '__dict__'):
            # If the reference is in the __dict__ of an object
            for key, value in referrer.__dict__.items():
                if value is old_obj:
                    setattr(referrer, key, new_obj)

def interleave_list(l1: List[mindspore.Tensor], l2: List[mindspore.Tensor]):
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


def unrotate(cache: mindspore.Tensor, seqlen: int) -> mindspore.Tensor:
    assert cache.ndim == 3  # (W, H, D)
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return ops.cat([cache[position:], cache[:position]], axis=0)

def index_copy(self, dim, index, tensor2):
    select = self.index_select(dim, index)
    index = index.astype(mindspore.int32)
    # ms.Tensor.index_add is an in-place operation, so we need to deepcopy input first
    # ms.ops.index_add supports only Parameter input so we use ms.Tensor.index_add here
    output0 = self.index_add(dim, index, select, alpha=-1)
    output = output0.index_add(dim, index, tensor2)
    return output

class CacheView:
    def __init__(self, cache_k: mindspore.Tensor, cache_v: mindspore.Tensor, metadata: RotatingCacheInputMetadata, kv_seqlens: mindspore.Tensor, layer_id):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata
        self.layer_id = layer_id

    def update(self, xk: mindspore.Tensor, xv: mindspore.Tensor):
        """
        to_cache_mask masks the last [sliding_window] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        origin_shape = self.cache_k.shape
        flat_cache_k = self.cache_k.view(origin_shape[0], -1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(origin_shape[0], -1, n_kv_heads, head_dim)
        # print(xk[self.metadata.to_cache_mask].shape, self.cache_k.shape)
        # print(self.metadata.cache_positions)

        flat_cache_k[self.layer_id, self.metadata.cache_positions] = xk[self.metadata.to_cache_mask]
        flat_cache_v[self.layer_id, self.metadata.cache_positions] = xv[self.metadata.to_cache_mask]
        # flat_cache_k = index_copy(flat_cache_k, 0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        # flat_cache_v = index_copy(flat_cache_v, 0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

        replace_references(self.cache_k, flat_cache_k.view(origin_shape))
        replace_references(self.cache_v, flat_cache_v.view(origin_shape))

    def interleave_kv(self, xk: mindspore.Tensor, xv: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3 # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(T, H, D)]
        xk = ops.split(xk, self.metadata.seqlens)
        xv = ops.split(xv, self.metadata.seqlens)
        assert len(xk) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Order elements in cache by position by unrotating
        cache_k = [unrotate(t, s) for t, s in zip(self.cache_k[self.layer_id], self.kv_seqlens)]
        cache_v = [unrotate(t, s) for t, s in zip(self.cache_v[self.layer_id], self.kv_seqlens)]

        interleaved_k = interleave_list(cache_k, xk)
        interleaved_v = interleave_list(cache_v, xv)

        return ops.cat(interleaved_k, axis=0), ops.cat(interleaved_v, axis=0)

    @property
    def sliding_window(self):
        return self.cache_k[self.layer_id].shape[1]

    @property
    def key(self) -> mindspore.Tensor:
        return self.cache_k[self.layer_id, :len(self.kv_seqlens)]

    @property
    def value(self) -> mindspore.Tensor:
        return self.cache_v[self.layer_id, :len(self.kv_seqlens)]

    @property
    def prefill(self):
        return self.metadata.prefill

    @property
    def mask(self):
        return self.metadata.mask


class RotatingBufferCache:
    """
    This is an example that implements a less naive rotating buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """
    def __init__(self, n_layers: int, max_batch_size: int, sliding_window: int, n_kv_heads: int, head_dim: int):

        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = ops.zeros((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        self.cache_v = ops.zeros((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        # holds the valid length for each batch element in the cache
        self.kv_seqlens = None

    def get_view(self, layer_id: int, metadata: RotatingCacheInputMetadata) -> CacheView:
        return CacheView(self.cache_k, self.cache_v, metadata, self.kv_seqlens, layer_id)

    def reset(self):
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int):
        self.kv_seqlens = ops.zeros((batch_size,), dtype=mindspore.int64)

    def to(self, dtype):
        self.cache_k = self.cache_k.to(dtype=dtype)
        self.cache_v = self.cache_v.to(dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]):
        self.kv_seqlens = self.kv_seqlens + mindspore.Tensor(seqlens, dtype=mindspore.int64)

    def get_input_metadata(self, seqlens: List[int]) -> RotatingCacheInputMetadata:
        """
            inpput = seqlens [5,7,2] // seqpos [0, 1, 3] // sliding_window 3
            --> only cache last 3 tokens in each sequence
            - to_cache_mask = [0 0 1 1 1 | 0 0 0 0 1 1 1 | 1 1]
            - cached_elements = [3 | 3 | 2]
            --> absolute positions are used for rope
            - positions = [0 1 2 3 4 | 1 2 3 4 5 6 7 | 3 4]
            --> cache positions are positions cache_masked, modulo sliding_window + batch_idx * sliding_window
            - cache_positions = [2 0 1 | 5 3 4 | 6 7]
        """
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))
        assert len(seqlens) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        seqpos = self.kv_seqlens.tolist()

        assert len(seqlens) > 0, seqlens
        masks = [
            [x >= seqlen - self.sliding_window for x in range(seqlen)]
            for seqlen in seqlens
        ]
        to_cache_mask = mindspore.Tensor(sum(masks, []), dtype=mindspore.bool_)
        cached_elements = mindspore.Tensor([sum(mask) for mask in masks], dtype=mindspore.int64)
        positions = ops.cat([ops.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]).to(dtype=mindspore.int64)
        batch_idx = mindspore.Tensor(sum([[i]*seqlen for i, seqlen in enumerate(seqlens)], []), dtype=mindspore.int64)
        cache_positions = positions % self.sliding_window + batch_idx * self.sliding_window

        first_prefill = seqpos[0] == 0
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)
        if first_prefill:
            assert all([pos == 0 for pos in seqpos]), (seqpos)
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(self.sliding_window)
        elif subsequent_prefill:
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=[s + cached_s.clamp(max=self.sliding_window).item() for (s, cached_s) in zip(seqlens, self.kv_seqlens)]
            ).make_local_attention_from_bottomright(self.sliding_window)
        else:
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=self.sliding_window,
                kv_seqlen=(self.kv_seqlens + cached_elements).clamp(max=self.sliding_window).tolist()
            )

        return RotatingCacheInputMetadata(
            positions=positions,
            to_cache_mask=to_cache_mask,
            cached_elements=cached_elements,
            cache_positions=cache_positions[to_cache_mask],
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )