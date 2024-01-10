# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union
import mindspore
from mindspore import ops


class AttentionBias:
    """Base class for a custom bias that can be applied \
        as the attn_bias argument in
        :attr:`xformers.ops.memory_efficient_attention`.

    That function has the ability to add a tensor, the
    attention bias, to the QK^T matrix before it is used
    in the softmax part of the attention calculation.
    The attention bias tensor with shape
    (B or 1, n_queries, number of keys)
    can be given as the attn_bias input.
    The most common use case is for an attention bias is
    to contain only zeros and negative infinities, which forms
    a mask so that some queries only attend to some keys.

    Children of this class define alternative things which can
    be used as the attn_bias input to define an attention bias which
    forms such a mask, for some common cases.

    When using an :attr:`xformers.ops.AttentionBias`
    instead of a :attr:`mindspore.Tensor`, the mask matrix does
    not need to be materialized, and can be
    hardcoded into some kernels for better performance.

    See:

    - :attr:`xformers.ops.fmha.attn_bias.LowerTriangularMask`
    - :attr:`xformers.ops.fmha.attn_bias.LowerTriangularFromBottomRightMask`
    - :attr:`xformers.ops.fmha.attn_bias.LowerTriangularMaskWithTensorBias`
    - :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`
    - :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`

    """

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        """
        Materializes the bias as a `mindspore.Tensor`. This is very slow
        and we don't attempt to make it fast. Only use for debugging/testing.

        Shape should be like `[*, q_seqlen, k_seqlen]`
        """
        raise NotImplementedError()


def _materialize_causal_mask(
    shape: Tuple[int, ...],
    dtype = mindspore.float32,
    *,
    window_size: Optional[int] = None,
    from_bottomright: bool = False,
) -> mindspore.Tensor:
    create_as = dtype if dtype is not mindspore.bfloat16 else mindspore.float32
    tensor = ops.full(  # type: ignore
        shape,
        dtype=create_as,
        fill_value=1,
    )

    num_queries, num_keys = shape[-2:]
    shift = 0
    if from_bottomright:
        shift = num_keys - num_queries

    mask = ops.tril(tensor, diagonal=shift).to(dtype)  # type: ignore
    if window_size is not None:
        mask = ops.triu(mask, diagonal=shift - window_size + 1)
    mask = ops.log(mask)
    return mask.to(dtype)


@dataclass
class LocalAttentionFromBottomRightMask(AttentionBias):
    """
    A local attention mask

    The query at position :math:`q` can attend the key at position :math:`k` if
    :math:`q - window\\_left <= k + s <= q + window\\_right`

    With :math:`s = num\\_queries - num\\_keys`

    :Example:

    .. code-block:: python

        import torch
        from xformers.ops import fmha

        bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=1, window_right=2)
        print(bias.materialize(shape=(4, 4)).exp())
        print(bias.materialize(shape=(4, 5)).exp())

    .. code-block:: text

        # 4x4
        tensor([[1., 1., 1., 0.],
                [1., 1., 1., 1.],
                [0., 1., 1., 1.],
                [0., 0., 1., 1.]])

        # 4x5
        tensor([[1., 1., 1., 1., 0.],
                [0., 1., 1., 1., 1.],
                [0., 0., 1., 1., 1.],
                [0., 0., 0., 1., 1.]])

    :Illustration:

    .. figure:: /_static/local_attn.png
        :width: 240px

        The total window size is :math:`window\\_left + 1 + window\\_right`
    """

    window_left: int
    window_right: int

    def __post_init__(self) -> None:
        if self.window_left < 0:
            raise ValueError(
                "Invalid window value passed to "
                "`LocalAttentionFromBottomRightMask`: expected"
                f"`window_left > 0` but got window_left={self.window_left}"
            )
        if self.window_right < 0:
            raise ValueError(
                "Invalid window value passed to "
                "`LocalAttentionFromBottomRightMask`: expected"
                f"`window_right > 0` but got window_right={self.window_right}"
            )

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        create_as = dtype if dtype is not mindspore.bfloat16 else mindspore.float32
        mask = ops.full(  # type: ignore
            shape,
            dtype=create_as,
            fill_value=1,
        )

        num_queries, num_keys = shape[-2:]
        shift = num_keys - num_queries

        mask = ops.triu(mask, diagonal=shift - self.window_left)
        mask = ops.tril(mask, diagonal=shift + self.window_right)
        mask = ops.log(mask)
        return mask.to(dtype)


class LowerTriangularMask(AttentionBias):
    """
    A lower-triangular (aka causal) mask

    A query Q cannot attend to a key which is farther from the
    initial key than Q is from the initial query.

    See also :attr:`LowerTriangularFromBottomRightMask` if the number
    of queries is not equal to the number of keys/values.
    """

    def __init__(self, *tensor_args, **tensor_kwargs) -> None:
        # NOTE: Unused arguments, we keep them for backward compatibility
        super().__init__()

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        return _materialize_causal_mask(shape, dtype=dtype)

    def add_bias(self, bias: mindspore.Tensor) -> "LowerTriangularMaskWithTensorBias":
        """
        Creates a new causal mask with an arbitrary ``mindspore.Tensor`` bias
        """
        return LowerTriangularMaskWithTensorBias(bias)


class LowerTriangularFromBottomRightMask(AttentionBias):
    """
    A causal masking.

    This mask is exactly the same as :attr:`LowerTriangularMask` when there is
    the same number of queries and keys.
    When the number of queries is different from the number of keys,
    it is a triangular mask shifted so that the last query can attend to
    the last key.
    In other words, a query Q cannot attend to a key which is nearer the
    final key than Q is to the final query.


    .. figure:: /_static/causal_bottom_right.png

        The difference between :attr:`LowerTriangularMask` (left) and
        :attr:`LowerTriangularFromBottomRightMask` (right). They become
        equivalent if the number of queries equals the number of keys.
    """

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        return _materialize_causal_mask(
            shape, dtype=dtype, from_bottomright=True
        )

    def make_local_attention(
        self, window_size: int
    ) -> "LowerTriangularFromBottomRightLocalAttentionMask":
        """
        Create a new bias which combines local + causal attention.

        See :attr:`LowerTriangularFromBottomRightLocalAttentionMask`
        """
        return LowerTriangularFromBottomRightLocalAttentionMask(window_size)


@dataclass
class LowerTriangularFromBottomRightLocalAttentionMask(
    LowerTriangularFromBottomRightMask
):
    """
    A mask that combines both :attr:`LowerTriangularFromBottomRightMask` and
    local attention.

    A query whose distance from the final query is X cannot attend to a key
    whose distance to the final key is either of:

    * less than X (i.e. "causal attention", same as :attr:`LowerTriangularFromBottomRightMask`)
    * greater than X + window_size (i.e. "local attention")


    .. figure:: /_static/causal_bottom_right_local.png

        The mask from :attr:`LowerTriangularFromBottomRightLocalAttentionMask`.
        The green area is calculated, and the grey area is masked out.
    """

    _window_size: int

    def __post_init__(self) -> None:
        if self._window_size <= 0:
            raise ValueError(
                f"Expected `window_size > 0`, but window_size={self._window_size}"
            )

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        return _materialize_causal_mask(
            shape,
            dtype=dtype,
            window_size=self._window_size,
            from_bottomright=True,
        )


class LowerTriangularMaskWithTensorBias(LowerTriangularMask):
    """A lower-triangular (aka causal) mask with an additive bias"""

    def __init__(self, bias: mindspore.Tensor) -> None:
        self._bias = bias

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        return super().materialize(shape, dtype=dtype) + self._bias


@dataclass
class _SeqLenInfo:
    """
    (Internal) Represents the division of a dimension into blocks.

    For example, to represents a dimension of length 7 divided into
    three blocks of lengths 2, 3 and 2, use `from_seqlength([2, 3, 2])`.
    The members will be:
        max_seqlen: 3
        min_seqlen: 2
        seqstart_py: [0, 2, 5, 7]
        seqstart: torch.IntTensor([0, 2, 5, 7])
    """

    seqstart: mindspore.Tensor
    max_seqlen: int
    min_seqlen: int
    seqstart_py: List[int]

    def intervals(self) -> Iterable[Tuple[int, int]]:
        yield from zip(self.seqstart_py, self.seqstart_py[1:])

    @classmethod
    def from_seqlens(cls, seqlens: Iterable[int]) -> "_SeqLenInfo":
        """
        Input tensors are assumed to be in shape [B, M, *]
        """
        assert not isinstance(seqlens, mindspore.Tensor)
        seqstart_py = [0]
        max_seqlen = -1
        min_seqlen = -1
        for seqlen in seqlens:
            min_seqlen = min(min_seqlen, seqlen) if min_seqlen != -1 else seqlen
            max_seqlen = max(max_seqlen, seqlen)
            seqstart_py.append(seqstart_py[len(seqstart_py) - 1] + seqlen)
        seqstart = mindspore.Tensor(seqstart_py, dtype=mindspore.int32)
        return cls(
            max_seqlen=max_seqlen,
            min_seqlen=min_seqlen,
            seqstart=seqstart,
            seqstart_py=seqstart_py,
        )

    def split(
        self, x: mindspore.Tensor, batch_sizes: Optional[Sequence[int]] = None
    ) -> List[mindspore.Tensor]:
        if self.seqstart_py[-1] != x.shape[1] or x.shape[0] != 1:
            raise ValueError(
                f"Invalid `mindspore.Tensor` of shape {x.shape}, expected format "
                f"(B, M, *) with B=1 and M={self.seqstart_py[-1]}\n"
                f" seqstart: {self.seqstart_py}"
            )
        if batch_sizes is None:
            batch_sizes = [1] * (len(self.seqstart_py) - 1)
        split_chunks = []
        it = 0
        for batch_size in batch_sizes:
            split_chunks.append(
                self.seqstart_py[it + batch_size] - self.seqstart_py[it]
            )
            it += batch_size
        return [
            tensor.reshape([bs, -1, *tensor.shape[2:]])
            for bs, tensor in zip(batch_sizes, x.split(split_chunks, axis=1))
        ]


@dataclass
class _PaddedSeqLenInfo(_SeqLenInfo):
    """
    (Internal)  Represents the division of a dimension into blocks which are
    padded out to the same total length.

    For example, to represent a dimension of length 12 with space for
    three blocks of length 4, but where the occupied lengths are
    2, 3 and 2, use `from_seqlens_padded([2, 3, 2], 4)`.

    The layout along the dimension is

     0 ─►  block 0
           block 0
           <space>
           <space>
     4 ─►  block 1
           block 1
           block 1
           <space>
     8 ─►  block 2
           block 2
           <space>
           <space>
    12 ─►

    The members will be:
        max_seqlen: 3
        min_seqlen: 2
        seqstart_py: [0, 4, 8, 12]
        seqstart: torch.IntTensor([0, 4, 8, 12])
        seqlen_py: [2, 3, 2]
        seqlen: torch.IntTensor([2, 3, 2])
        padding: 4
    """

    seqlen: mindspore.Tensor
    seqlen_py: Sequence[int]
    padding: int
    # From parent: seqstart[i] contains the start position
    # of the i-th sequence
    # seqstart: mindspore.Tensor

    def __post_init__(self) -> None:
        assert len(self.seqstart_py) == len(self.seqlen_py) + 1

    def intervals(self) -> Iterable[Tuple[int, int]]:
        for (start, _), length in zip(super().intervals(), self.seqlen_py):
            yield start, start + length

    @classmethod
    def from_seqlens(cls, seqlens: Iterable[int]) -> "_SeqLenInfo":
        raise RuntimeError(
            "Use either `_SeqLenInfo.from_seqlens` or `_PaddedSeqLenInfo.from_seqlens_padded`"
        )

    @classmethod
    def from_seqlens_padded(
        cls, seqlens: Sequence[int], padding: int
    ) -> "_PaddedSeqLenInfo":
        """
        Input tensors are assumed to be in shape [B, M, *]
        seqstart = padding * torch.arange(batch_size)
        """
        assert not isinstance(seqlens, mindspore.Tensor)
        assert all(seqlen <= padding for seqlen in seqlens)
        seqstart_py = list(range(0, len(seqlens) * padding + 1, padding))
        return cls(
            seqlen=mindspore.Tensor(seqlens, dtype=mindspore.int32),
            seqlen_py=seqlens,
            max_seqlen=max(seqlens),
            min_seqlen=min(seqlens),
            seqstart=mindspore.Tensor(seqstart_py, dtype=mindspore.int32),
            seqstart_py=seqstart_py,
            padding=padding,
        )

    def split(
        self, x: mindspore.Tensor, batch_sizes: Optional[Sequence[int]] = None
    ) -> List[mindspore.Tensor]:
        raise NotImplementedError("_PaddedSeqLenInfo.split")


@dataclass
class BlockDiagonalMask(AttentionBias):
    """
    A block-diagonal mask that can be passed as ``attn_bias``
    argument to :attr:`xformers.ops.memory_efficient_attention`.

    Queries and Keys are each divided into the same number of blocks.
    Queries in block i only attend to keys in block i.

    .. figure:: /_static/block_diag_bias.png

        This bias can be used to handle a batch of sequences of
        different lengths, via :attr:`BlockDiagonalMask.from_tensor_list`

    :Example:

    .. code-block:: python

        import torch
        from xformers.ops import fmha

        K = 16
        dtype = torch.float16
        device = "cuda"
        list_x = [
            torch.randn([1, 3, 1, K], dtype=dtype, device=device),
            torch.randn([1, 6, 1, K], dtype=dtype, device=device),
            torch.randn([1, 2, 1, K], dtype=dtype, device=device),
        ]
        attn_bias, x = fmha.BlockDiagonalMask.from_tensor_list(list_x)
        linear = torch.nn.Linear(K, K * 3).to(device=device, dtype=dtype)

        q, k, v = linear(x).reshape([1, -1, 1, 3, K]).unbind(-2)
        out = fmha.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        list_out = attn_bias.split(out)
        print(list_out[0].shape)  # [1, 3, 1, K]
        assert tuple(list_out[0].shape) == (1, 3, 1, K)

    """

    q_seqinfo: _SeqLenInfo
    k_seqinfo: _SeqLenInfo
    _batch_sizes: Optional[Sequence[int]] = None

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        return ops.zeros(
            shape,
            dtype=dtype,
        )

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        # assert shape[-1] == self.k_seqinfo.seqstart_py[-1], (
        #     shape[-1],
        #     self.k_seqinfo.seqstart_py[-1],
        # )
        # assert shape[-2] == self.q_seqinfo.seqstart_py[-1], (
        #     shape[-2],
        #     self.q_seqinfo.seqstart_py[-1],
        # )
        mask = ops.fill(dtype, shape[-2:], -math.inf)
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(
            zip(
                self.q_seqinfo.intervals(),
                self.k_seqinfo.intervals(),
            )
        ):
            mask[q_start:q_end, k_start:k_end] = self._create_block_mask(
                (q_end - q_start, k_end - k_start),
                dtype=dtype,
            )
        for _ in range(len(shape) - 2):
            mask = mask.unsqueeze(0)
        return mask.broadcast_to(shape)

    @classmethod
    def from_seqlens(
        cls,
        q_seqlen: Sequence[int],
        kv_seqlen: Optional[Sequence[int]] = None,
    ) -> "BlockDiagonalMask":
        """Creates a :attr:`BlockDiagonalMask` from a list of tensors lengths for query and key/value.

        Args:
            q_seqlen (Union[Sequence[int], mindspore.Tensor]): List or tensor of sequence lengths for query tensors
            kv_seqlen (Union[Sequence[int], mindspore.Tensor], optional): List or tensor of sequence lengths for key/value.
                    (Defaults to ``q_seqlen``.)
        Returns:
            BlockDiagonalMask
        """
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen)
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen)
        if kv_seqlen is None or q_seqlen == kv_seqlen:
            k_seqinfo = q_seqinfo
        else:
            k_seqinfo = _SeqLenInfo.from_seqlens(kv_seqlen)
        return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)

    @classmethod
    def from_tensor_list(
        cls,
        tensors: Sequence[mindspore.Tensor],
    ) -> Tuple["BlockDiagonalMask", mindspore.Tensor]:
        """Creates a :attr:`BlockDiagonalMask` from a list of tensors, and returns the tensors
        concatenated on the sequence length dimension

        .. figure:: /_static/block_diag_cat_split.png

            See also :attr:`BlockDiagonalMask.split` to split the returned
            :attr:`mindspore.Tensor` back to a list of tensors of varying sequence length

        Args:
            tensors (Sequence[mindspore.Tensor]): A list of tensors of shape ``[B, M_i, *]``.
                All tensors should have the same dimension and the same batch size ``B``, but
                they can have different sequence length ``M``.

        Returns:
            Tuple[BlockDiagonalMask, mindspore.Tensor]: The corresponding bias for the attention
            along with `tensors` concatenated on the sequence length dimension, with shape ``[1, sum_i{M_i}, *]``
        """
        batch_sizes = [tensor.shape[0] for tensor in tensors]
        seqlens = []
        for x in tensors:
            for _ in range(x.shape[0]):
                seqlens.append(x.shape[1])
        block_diag = cls.from_seqlens(seqlens)
        block_diag._batch_sizes = batch_sizes
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in tensors)
        concat_tensors = ops.cat(tensors_bs1, axis=1)
        return block_diag, concat_tensors

    @classmethod
    def from_tensor_lists_qkv(
        cls,
        tensors_q: Sequence[mindspore.Tensor],
        tensors_k: Sequence[mindspore.Tensor],
        tensors_v: Optional[Sequence[mindspore.Tensor]] = None,
    ) -> Tuple["BlockDiagonalMask", mindspore.Tensor, mindspore.Tensor, Optional[mindspore.Tensor]]:
        assert len(tensors_q) == len(tensors_k)
        assert tensors_v is None or len(tensors_v) == len(tensors_q)
        batch_sizes = [tensor.shape[0] for tensor in tensors_q]
        q_seqlens, kv_seqlens = [], []
        for i, (q, k) in enumerate(zip(tensors_q, tensors_k)):
            assert q.shape[0] == k.shape[0]
            q_seqlens += [q.shape[1]] * q.shape[0]
            kv_seqlens += [k.shape[1]] * k.shape[0]
            assert tensors_v is None or tensors_v[i].shape[:2] == k.shape[:2]
        block_diag = cls.from_seqlens(q_seqlens, kv_seqlens)
        block_diag._batch_sizes = batch_sizes
        return (
            block_diag,
            ops.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_q], axis=1),
            ops.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_k], axis=1),
            ops.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_v], axis=1)
            if tensors_v is not None
            else None,
        )

    def split_queries(self, tensor: mindspore.Tensor) -> Sequence[mindspore.Tensor]:
        return self.q_seqinfo.split(tensor, self._batch_sizes)

    def split_kv(self, tensor: mindspore.Tensor) -> Sequence[mindspore.Tensor]:
        return self.k_seqinfo.split(tensor, self._batch_sizes)

    def split(self, tensor: mindspore.Tensor) -> Sequence[mindspore.Tensor]:
        """The inverse operation of :attr:`BlockDiagonalCausalMask.from_tensor_list`

        Args:
            tensor (mindspore.Tensor): Tensor of tokens of shape ``[1, sum_i{M_i}, *]``

        Returns:
            Sequence[mindspore.Tensor]: A list of tokens with possibly different sequence lengths
        """
        assert self.q_seqinfo is self.k_seqinfo
        return self.q_seqinfo.split(tensor, self._batch_sizes)

    def make_causal(self) -> "BlockDiagonalCausalMask":
        """Makes each block causal"""
        return BlockDiagonalCausalMask(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=self.k_seqinfo,
            _batch_sizes=self._batch_sizes,
        )

    def make_causal_from_bottomright(self) -> "BlockDiagonalCausalFromBottomRightMask":
        """Makes each block causal with a possible non-causal prefix"""
        return BlockDiagonalCausalFromBottomRightMask(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=self.k_seqinfo,
            _batch_sizes=self._batch_sizes,
        )

    def make_local_attention(
        self, window_size: int
    ) -> "BlockDiagonalCausalLocalAttentionMask":
        """Experimental: Makes each block causal with local attention"""
        return BlockDiagonalCausalLocalAttentionMask(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=self.k_seqinfo,
            _batch_sizes=self._batch_sizes,
            _window_size=window_size,
        )

    def make_local_attention_from_bottomright(
        self, window_size: int
    ) -> "BlockDiagonalCausalLocalAttentionFromBottomRightMask":
        """Experimental: Makes each block causal with local attention, start from bottom right"""
        return BlockDiagonalCausalLocalAttentionFromBottomRightMask(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=self.k_seqinfo,
            _batch_sizes=self._batch_sizes,
            _window_size=window_size,
        )


@dataclass
class BlockDiagonalCausalMask(BlockDiagonalMask):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`, except that each block is causal.

    Queries and Keys are each divided into the same number of blocks.
    A query Q in block i cannot attend to a key which is not in block i,
    nor one which is farther from the initial key in block i than Q
    is from the initial query in block i.
    """

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        return LowerTriangularMask().materialize(
            shape,
            dtype=dtype,
        )


@dataclass
class BlockDiagonalCausalFromBottomRightMask(BlockDiagonalMask):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`, except that each block is causal.
    This mask allows for a non-causal prefix
    NOTE: Each block should have `num_keys >= num_queries` otherwise the forward pass is not
    defined (softmax of vector of `-inf` in the attention)

    Queries and keys are each divided into the same number of blocks.
    A query Q in block i cannot attend to a key which is not in block i,
    nor one which nearer the final key in block i than Q is to the
    final query in block i.
    """

    def __post_init__(self) -> None:
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(
            zip(
                self.q_seqinfo.intervals(),
                self.k_seqinfo.intervals(),
            )
        ):
            num_queries = q_end - q_start
            num_keys = k_end - k_start
            if num_keys < num_queries:
                raise ValueError(
                    f"Block #{i} has num_keys={num_keys} and num_queries={num_queries}."
                    " Expected `num_keys >= num_queries`"
                )

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        return LowerTriangularFromBottomRightMask().materialize(
            shape=shape, dtype=dtype
        )


@dataclass
class BlockDiagonalCausalWithOffsetPaddedKeysMask(AttentionBias):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`,
    except an offset on causality is allowed for each block and we support padding for k/v

    The keys and values are divided into blocks which are padded out to
    the same total length.
    For example, if there is space for 12 keys, for three blocks of
    max length 4, but we only want to use the first 2, 3 and 2
    of each block, use `kv_padding=4` and `kv_seqlens=[2, 3, 2]`.
    The queries are divided into blocks, without padding, of lengths given by
    q_seqlen.

    A query Q in block i cannot attend to a key which is not in block i,
    nor one which is not in use (i.e. in the padded area),
    nor one which is nearer to the final key in block i
    than Q is to the final query in block i.
    """

    q_seqinfo: _SeqLenInfo
    k_seqinfo: _PaddedSeqLenInfo
    causal_diagonal: Any = None  # unused. Exists for BC only.

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        return LowerTriangularFromBottomRightMask().materialize(
            shape=shape, dtype=dtype
        )

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        # if shape[-1] != self.k_seqinfo.seqstart_py[-1]:
        #     raise ValueError("k shapes wrong")
        # if shape[-2] != self.q_seqinfo.seqstart_py[-1]:
        #     raise ValueError("q shapes wrong")
        mask = ops.fill(dtype, shape[-2:], -math.inf)

        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(
            zip(
                self.q_seqinfo.intervals(),
                self.k_seqinfo.intervals(),
            )
        ):
            mask[q_start:q_end, k_start:k_end] = self._create_block_mask(
                (q_end - q_start, k_end - k_start),
                dtype=dtype,
            )
        for _ in range(len(shape) - 2):
            mask = mask.unsqueeze(0)
        return mask.broadcast_to(shape)

    @classmethod
    def from_seqlens(
        cls,
        q_seqlen: Sequence[int],
        kv_padding: int,
        kv_seqlen: Sequence[int],
        causal_diagonal: Any = None,
    ) -> "BlockDiagonalCausalWithOffsetPaddedKeysMask":
        """Creates a :attr:`BlockDiagonalCausalWithOffsetPaddedKeysMask` from a list of tensor
        lengths for query and key/value.

        Args:
            q_seqlen (Sequence[int]): List or tensor of sequence lengths for query tensors
            kv_padding (int): Padding for k/v - also an upperbound on each individual key length
            kv_seqlen (Sequence[int]): List or tensor of sequence lengths for key/value.
            causal_diagonal: unused, for BC only
        Returns:
            BlockDiagonalCausalWithOffsetPaddedKeysMask
        """
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen), (
            q_seqlen,
            kv_seqlen,
        )
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen)
        k_seqinfo = _PaddedSeqLenInfo.from_seqlens_padded(kv_seqlen, kv_padding)
        return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)


@dataclass
class BlockDiagonalCausalLocalAttentionMask(BlockDiagonalCausalMask):
    """
    (Experimental feature)
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`.
    This makes the mask "local" and the attention pattern banded.

    Query i only attends to keys in its block and cannot attend keys further than "window_size"
    from it.
    """

    _window_size: int = 0  # forced due to inheritance and default arguments

    def __post_init__(self):
        if self._window_size <= 0:
            raise ValueError(
                f"Expected `window_size > 0`, but window_size={self._window_size}"
            )
        q_seqlen = [
            y - x
            for x, y in zip(
                self.q_seqinfo.seqstart_py[:-1], self.q_seqinfo.seqstart_py[1:]
            )
        ]
        kv_seqlen = [
            y - x
            for x, y in zip(
                self.k_seqinfo.seqstart_py[:-1], self.k_seqinfo.seqstart_py[1:]
            )
        ]
        for q, k in zip(q_seqlen, kv_seqlen):
            if q - self._window_size >= k:
                # Each query only attends to keys no further than window_size back.
                # When q > k + window_size, there will be a query for which the window doesn't reach any key.
                raise RuntimeError(
                    f"No keys are attended in q_seqlen {q} k_seqlen {k} with sliding window {self._window_size}"
                )

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        return _materialize_causal_mask(
            shape,
            dtype=dtype,
            window_size=self._window_size,
        )


@dataclass
class BlockDiagonalCausalLocalAttentionFromBottomRightMask(
    BlockDiagonalCausalFromBottomRightMask
):
    """
    (Experimental feature)
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`.
    This makes the mask "local" and the attention pattern banded.

    Query i only attends to keys in its block and cannot attend keys further than "window_size"
    from it.
    """

    _window_size: int = 0  # forced due to inheritance and default arguments

    def __post_init__(self):
        super().__post_init__()
        if self._window_size <= 0:
            raise ValueError(
                f"Expected `window_size > 0`, but window_size={self._window_size}"
            )

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype = mindspore.float32,
    ) -> mindspore.Tensor:
        return _materialize_causal_mask(
            shape,
            dtype=dtype,
            window_size=self._window_size,
            from_bottomright=True,
        )
