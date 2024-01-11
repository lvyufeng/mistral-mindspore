import json
import logging
import types
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

import mindspore
from mindspore import ops, nn, Parameter
from mindspore._c_expression import Tensor as Tensor_
from mindspore.common._stub_tensor import StubTensor
from mindspore.common.initializer import initializer
from mindspore.ops.operations._inner_ops import Send, Receive
from mindspore.ops._primitive_cache import _get_cache_prim

from simple_parsing.helpers import Serializable

from mistral.rope import precompute_freqs_cis, apply_rotary_emb
from mistral.cache import CacheView, RotatingBufferCache
from mistral.moe import MoeArgs, MoeLayer
from mistral.ckpt_reader import load
from mistral.attn_bias import AttentionBias

from mistral.attention import ref_attention

try:
    from mindspore.nn.layer.flash_attention import FlashAttention
    FLASHATTENTION_IMPORT_VALID = True
except ImportError:
    FLASHATTENTION_IMPORT_VALID = False

is_ascend = mindspore.get_context('device_target') == 'Ascend'
world_group = 'hccl_world_group' if is_ascend else 'nccl_world_group'

@dataclass
class ModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int

    max_batch_size: int = 0

    # For rotary embeddings. If not set, will be infered from sliding window.
    rope_theta: Optional[float] = None
    # If this is set, use sliding window attention rotating cache.
    sliding_window: Optional[int] = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe: Optional[MoeArgs] = None


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: mindspore.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int]) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=ops.cat([ops.arange(0, seqlen) for seqlen in seqlens]).to(
                dtype=mindspore.int64
            )
        )


def repeat_kv(keys: mindspore.Tensor, values: mindspore.Tensor, repeats: int, dim: int):
    keys = ops.repeat_interleave(keys, repeats=repeats, axis=dim)
    values = ops.repeat_interleave(values, repeats=repeats, axis=dim)
    return keys, values

class Embedding(nn.Cell):
    """patched Embedding"""
    def __init__(self, vocab_size, embedding_size, padding_idx=None, use_one_hot=False, dtype=mindspore.float16, weight_init='zeros'):
        """Initialize Embedding."""
        super().__init__()
        self.use_one_hot = use_one_hot
        self.dtype = dtype
        self.padding_idx = padding_idx
        self.embedding_size = embedding_size
        self.weight = Parameter(initializer(weight_init, [vocab_size, embedding_size], dtype=dtype), name='weight')

    def construct(self, ids):
        out_shape = ids.shape + (self.embedding_size,)
        flat_ids = ids.reshape((-1,))

        if self.use_one_hot:
            one_hot_ids = ops.one_hot(flat_ids, self.vocab_size)
            output_for_reshape = ops.matmul(one_hot_ids, self.weight)
        else:
            output_for_reshape = ops.gather(self.weight, flat_ids, 0)

        output = output_for_reshape.reshape(out_shape)
        return output

class Dense(nn.Cell):
    """patched Dense"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 has_bias=True,
                 weight_init='zeros',
                 bias_init='zeros',
                 dtype=mindspore.float32):
        """Initialize Dense."""
        super().__init__()
        # self.weight = Parameter(initializer(
        #     weight_init, [out_channels, in_channels], dtype=dtype), name="weight")
        self.weight = Parameter(Tensor_(shape=[out_channels, in_channels], dtype=dtype), name="weight")

        self.bias = None
        if has_bias:
            self.bias = Parameter(Tensor_(shape=[out_channels], dtype=dtype), name="bias")

    def construct(self, x):
        dense_ = _get_cache_prim(ops.Dense)()
        return dense_(x, self.weight, self.bias)

class Attention(nn.Cell):
    def __init__(self, args: ModelArgs, dtype=mindspore.float16):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = Dense(args.dim, args.n_heads * args.head_dim, has_bias=False, dtype=dtype)
        self.wk = Dense(args.dim, args.n_kv_heads * args.head_dim, has_bias=False, dtype=dtype)
        self.wv = Dense(args.dim, args.n_kv_heads * args.head_dim, has_bias=False, dtype=dtype)
        self.wo = Dense(args.n_heads * args.head_dim, args.dim, has_bias=False, dtype=dtype)

        if is_ascend and FLASHATTENTION_IMPORT_VALID:
            self.flash_attention = FlashAttention(head_dim=args.head_dim,
                                                  head_num=args.n_heads,
                                                  prev_block_num=65536,
                                                  next_block_num=0,
                                                  high_precision=True)

    def construct(
        self,
        x: mindspore.Tensor,
        freqs_cis: mindspore.Tensor,
        cache: Optional[CacheView],
    ) -> mindspore.Tensor:
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(
                seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim
            )

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        if hasattr(self, 'flash_attention'):
            origin_len = xq.shape[1]
            if xq.shape[1] % 16 != 0:
                xq = ops.pad(xq, (0, 0, 0, 0, 0, 16 - xq.shape[1] % 16), value=0)
            if key.shape[1] % 16 != 0:
                key = ops.pad(key, (0, 0, 0, 0, 0, 16 - key.shape[1] % 16), value=0)
            if val.shape[1] % 16 != 0:
                val = ops.pad(val, (0, 0, 0, 0, 0, 16 - val.shape[1] % 16), value=0)
            if cache is None:
                attn_bias = None
            else:
                if isinstance(cache.mask, AttentionBias):
                    attn_bias = (
                        cache.mask.materialize((xq.shape[0], 1, xq.shape[1], key.shape[1]))
                        .to(xq.dtype)
                        .squeeze()
                    )
                    d_min = float(np.finfo(mindspore.dtype_to_nptype(attn_bias.dtype)).min)
                    attn_bias = attn_bias.masked_fill(attn_bias == d_min, 1.0)
                else:
                    attn_bias = cache.mask
            output = self.flash_attention(xq.swapaxes(1, 2), key.swapaxes(1, 2), val.swapaxes(1, 2), attn_bias)
            output = output.swapaxes(1, 2)[:, :origin_len]
            # output = ref_attention(
            #     xq, key, val, None if cache is None else cache.mask
            # )
            # output = output[:, :origin_len]
        else:
            output = ref_attention(
                xq, key, val, None if cache is None else cache.mask
            )

        return self.wo(output.view(seqlen_sum, self.n_heads * self.head_dim))


class FeedForward(nn.Cell):
    def __init__(self, args: ModelArgs, dtype=mindspore.float16):
        super().__init__()

        self.w1 = Dense(args.dim, args.hidden_dim, has_bias=False, dtype=dtype)
        self.w2 = Dense(args.hidden_dim, args.dim, has_bias=False, dtype=dtype)
        self.w3 = Dense(args.dim, args.hidden_dim, has_bias=False, dtype=dtype)

    def construct(self, x) -> mindspore.Tensor:
        return self.w2(ops.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Cell):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=mindspore.float16):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ops.ones(dim, dtype=dtype))

    def _norm(self, x):
        return x * ops.rsqrt(x.pow(2).mean(-1, keep_dims=True) + self.eps)

    def construct(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Cell):
    def __init__(self, args: ModelArgs, dtype=mindspore.float16):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args, dtype=dtype)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)
        self.args = args

        self.feed_forward: nn.Cell
        if args.moe is not None:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(args=args, dtype=dtype) for _ in range(args.moe.num_experts)],
                gate=Dense(args.dim, args.moe.num_experts, has_bias=False, dtype=dtype),
                moe_args=args.moe,
            )
        else:
            self.feed_forward = FeedForward(args=args, dtype=dtype)

    def construct(
        self, x: mindspore.Tensor, freqs_cis: mindspore.Tensor, cache: Optional[CacheView]
    ) -> mindspore.Tensor:
        r = self.attention(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Cell):
    def __init__(
        self,
        args: ModelArgs,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
        dtype = mindspore.float16
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: Optional[mindspore.Tensor] = None
        assert self.vocab_size > 0
        assert pipeline_rank < num_pipeline_ranks, (pipeline_rank, num_pipeline_ranks)
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        # Modules specific to some ranks:
        self.tok_embeddings: Optional[Embedding] = None
        self.norm: Optional[RMSNorm] = None
        self.output: Optional[Dense] = None
        if pipeline_rank == 0:
            self.tok_embeddings = Embedding(args.vocab_size, args.dim, dtype=dtype)
        if pipeline_rank == num_pipeline_ranks - 1:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)
            self.output = Dense(args.dim, args.vocab_size, has_bias=False, dtype=dtype)
        # Initialize all layers but slice off those not of this rank.
        layers = [TransformerBlock(args=args, dtype=dtype) for _ in range(args.n_layers)]
        num_layers_per_rank = math.ceil(self.n_layers / self.num_pipeline_ranks)
        offset = self.pipeline_rank * num_layers_per_rank
        end = min(self.n_layers, offset + num_layers_per_rank)
        self.layers = nn.CellDict({str(i): layers[i] for i in range(offset, end)})
        self.n_local_layers = len(self.layers)

    @property
    def dtype(self):
        return next(self.get_parameters()).dtype

    @property
    def freqs_cis(self) -> mindspore.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's  dtype means we cannot register it as a buffer
        if self._precomputed_freqs_cis is None:
            # If no sliding window, assume a larger seqlen
            theta = self.args.rope_theta
            if theta is None:
                theta = 1000000.0 if self.args.sliding_window is None else 10000.0
            # theta = 10000.
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta
            )
        return self._precomputed_freqs_cis

    def construct_partial(
        self,
        input_ids: mindspore.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> mindspore.Tensor:
        """Local forward pass.

        If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        assert (
            len(seqlens) <= self.args.max_batch_size
        ), f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
        (num_toks,) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)
        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
        else:
            input_metadata = SimpleInputMetadata.from_seqlens(seqlens)

        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            h = self.tok_embeddings(input_ids)
        else:
            recv = _get_cache_prim(Receive)(sr_tag=0, src_rank=self.pipeline_rank - 1,
                        shape=[num_toks, self.args.dim], dtype=self.dtype, group=world_group)
            if is_ascend:
                depend = mindspore.tensor(0, self.dtype)
                h = recv(depend)
            else:
                h = recv()

        freqs_cis = self.freqs_cis[input_metadata.positions]
        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                assert input_metadata is not None
                cache_view = cache.get_view(local_layer_id, input_metadata)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            send = _get_cache_prim(Send)(sr_tag=0, dest_rank=self.pipeline_rank + 1, group=world_group)
            send(h)
            return h
        else:
            # Last rank has a final normalization step.
            assert self.norm is not None
            return self.norm(h)

    def construct(
        self,
        input_ids: mindspore.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> mindspore.Tensor:
        h = self.construct_partial(input_ids, seqlens, cache=cache)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # ignore the intermediate activations as we'll get the final output from
            # the last stage
            outs = ops.zeros((h.shape[0], self.vocab_size), h.dtype)
        else:
            assert self.output is not None
            outs = self.output(h)
        if self.num_pipeline_ranks > 1:
            if not is_ascend:
                broadcast = _get_cache_prim(ops.Broadcast)(self.num_pipeline_ranks - 1, group=world_group)
                outs, = broadcast((outs,))
            else:
                # broadcast op on Ascend cause error, use send/recv instead
                if self.pipeline_rank == self.num_pipeline_ranks - 1:
                    send = _get_cache_prim(Send)(sr_tag=1, dest_rank=0, group=world_group)
                    send(outs)
                elif self.pipeline_rank == 0:
                    recv = _get_cache_prim(Receive)(sr_tag=1, src_rank=self.num_pipeline_ranks - 1,
                                shape=outs.shape, dtype=outs.dtype, group=world_group)
                    outs = recv(outs)
        return outs.float()

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_to_load = {}
        skipped = set([])
        for k, v in state_dict.items():
            if k.startswith("tok_embeddings"):
                if self.pipeline_rank == 0:
                    state_to_load[k] = Parameter(Tensor_.from_numpy(v.astype(np.float16, copy=False)))
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            elif k.startswith("norm") or k.startswith("output"):
                if self.pipeline_rank == self.num_pipeline_ranks - 1:
                    state_to_load[k] = Parameter(Tensor_.from_numpy(v.astype(np.float16, copy=False)))
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            elif k.startswith("layers"):
                layer_id = k.split(".")[1]
                if layer_id in self.layers:
                    state_to_load[k] = Parameter(Tensor_.from_numpy(v.astype(np.float16, copy=False)))
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            else:
                raise ValueError(f"Unexpected key {k}")
        assert len(set(state_dict.keys())) == len(skipped.union(set(state_to_load.keys())))
        mindspore.load_param_into_net(self, state_to_load)

    @staticmethod
    def from_folder(
        folder: Path,
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        dtype=mindspore.float16,
    ) -> "Transformer":
        with open(folder / "params.json", "r") as f:
            model_args = ModelArgs.from_dict(json.load(f))
        model_args.max_batch_size = max_batch_size
        if num_pipeline_ranks > 1:
            pipeline_rank = mindspore.communication.get_rank()
        else:
            pipeline_rank = 0
        model = Transformer(
            model_args,
            pipeline_rank=pipeline_rank,
            num_pipeline_ranks=num_pipeline_ranks,
            dtype=dtype
        )
        loaded = load(str(folder / "consolidated.00.pth"), mmap=True)
        model.load_state_dict(loaded)
        return model
