import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from typing import Tuple

def view_as_complex(_x: mindspore.Tensor):
    '''
    view_as_complex
    '''
    _complex = _get_cache_prim(ops.Complex)()
    return _complex(_x[:,:,:,0], _x[:,:,:,1])


def precompute_freqs_cis(dim: int, end: int, theta: float) -> mindspore.Tensor:
    freqs = 1.0 / (theta ** (ops.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = ops.arange(end)  # type: ignore
    freqs = ops.outer(t, freqs).float()  # type: ignore
    return ops.polar(ops.ones_like(freqs, dtype=mindspore.float32), freqs)  # complex64


def apply_rotary_emb(
    xq: mindspore.Tensor,
    xk: mindspore.Tensor,
    freqs_cis: mindspore.Tensor,
) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
    xq_ = view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = ops.flatten(ops.view_as_real(xq_ * freqs_cis), start_dim=2)
    xk_out = ops.flatten(ops.view_as_real(xk_ * freqs_cis), start_dim=2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
