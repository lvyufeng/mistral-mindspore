import numpy as np
import mindspore
from mindspore import ops
from .attn_bias import AttentionBias

def ref_attention_bmk(q, k, v, attn_bias=None, p=0.0):
    if isinstance(attn_bias, AttentionBias):
        attn_bias = (
            attn_bias.materialize((q.shape[0], 1, q.shape[1], k.shape[1]))
            .to(q.dtype)
            .squeeze(1)
        )
    q = q * (1.0 / q.shape[-1] ** 0.5)
    if attn_bias is None:
        attn = q @ k.swapaxes(-2, -1)
    else:
        # equivalent to (q @ k.transpose(-2, -1) + m).softmax(-1) @ v
        # but faster, and is what is used in PyTorch now
        attn = ops.baddbmm(attn_bias, q, k.swapaxes(-2, -1))
    attn = ops.softmax(attn.float(), -1).type_as(q)
    if p > 0:
        attn = ops.dropout(attn, p=p)
    return attn @ v


def ref_attention(q, k, v, attn_bias, p=0.0):
    assert q.ndim == 4
    B, M, H, K = q.shape

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    if isinstance(attn_bias, mindspore.Tensor):
        attn_bias = attn_bias.reshape(B * H, M, M)
    out = ref_attention_bmk(T(q), T(k), T(v), attn_bias, p)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))
