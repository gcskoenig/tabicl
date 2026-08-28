from __future__ import annotations
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .rope import RotaryEmbedding
from .kv_cache import KVCacheEntry

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn3

    HAS_FLASH_ATTN3 = True
except ImportError:
    HAS_FLASH_ATTN3 = False

_use_flash_attn3 = True


@contextmanager
def flash_attn3_toggle(enabled: bool):
    """Context manager to temporarily enable or disable Flash Attention 3.

    Used by ``InferenceManager._run_forward()`` in ``inference.py`` to control
    whether Flash Attention 3 is used during each forward pass based on the
    ``use_fa3`` configuration.
    """
    global _use_flash_attn3
    old = _use_flash_attn3
    _use_flash_attn3 = enabled
    try:
        yield
    finally:
        _use_flash_attn3 = old


def set_flash_attn3_enabled(enabled: bool):
    """Globally enable or disable Flash Attention 3 for this process.

    Used by the pre-training ``Trainer`` (``--use_flash_attn3``): FA3 runs attention
    in fp16, so the TabICLv2 recipe enables it only for stages 2 and 3.
    """
    global _use_flash_attn3
    _use_flash_attn3 = enabled


def sdpa_with_flattened_batch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    ssmax_layer: Optional[nn.Module] = None,
) -> Tensor:
    """Applies scaled dot-product attention with flattened batch dimensions.

    This function handles arbitrary batch dimensions by flattening them before
    applying PyTorch's ``scaled_dot_product_attention`` and then reshaping the
    output back to the original shape. This flattening is necessary to properly
    trigger Flash Attention.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape (..., nh, tgt_len, hs) where:

        - ... represents arbitrary batch dimensions
        - nh is the number of attention heads
        - tgt_len is the target sequence length
        - hs is the head size (embedding dimension per head)

    k : Tensor
        Key tensor of shape (..., nh, src_len, hs) with matching batch dimensions.

    v : Tensor
        Value tensor of shape (..., nh, src_len, hs) with matching batch dimensions.

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (..., nh, tgt_len, src_len).

    dropout_p : float, default=0.0
        Dropout probability applied to attention weights.

    ssmax_layer : Optional[nn.Module], default=None
        If provided, applies scalable softmax (SSMax) scaling to queries before
        attention computation.

    Returns
    -------
    Tensor
        Attention output tensor of shape (..., nh, tgt_len, hs) preserving the
        original batch dimensions of the input.
    """

    q_shape = q.shape
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])
    if attn_mask is not None:
        attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])

    if ssmax_layer is not None:
        src_len = k.size(-2)
        q = ssmax_layer(q, src_len)

    # FlashAttn3 doesn't support dropout and custom attention mask
    if HAS_FLASH_ATTN3 and _use_flash_attn3 and q.is_cuda and attn_mask is None and dropout_p == 0.0:
        # FlashAttention only supports fp16, bf16, and fp8_e4m3
        # Convert to bf16 if needed, then convert back to original dtype
        orig_dtype = q.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            fa_dtype = torch.float16
        else:
            fa_dtype = orig_dtype

        flat_bs, nheads, seqlen_q, headdim = q.shape
        seqlen_k = k.shape[-2]
        q_fa = q.transpose(1, 2).reshape(flat_bs * seqlen_q, nheads, headdim).contiguous().to(fa_dtype)
        k_fa = k.transpose(1, 2).reshape(flat_bs * seqlen_k, nheads, headdim).contiguous().to(fa_dtype)
        v_fa = v.transpose(1, 2).reshape(flat_bs * seqlen_k, nheads, headdim).contiguous().to(fa_dtype)
        cu_seqlens_q = torch.arange(0, (flat_bs + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.arange(0, (flat_bs + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=q.device)
        out = flash_attn3(q_fa, k_fa, v_fa, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k)
        out = out.view(flat_bs, seqlen_q, nheads, headdim).transpose(1, 2).to(orig_dtype)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)

    return out.view(q_shape)


def multi_head_attention_forward(
    query: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    key: Optional[Tensor] = None,
    value: Optional[Tensor] = None,
    cached_kv: Optional[KVCacheEntry] = None,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    rope: Optional[RotaryEmbedding] = None,
    ssmax_layer: Optional[nn.Module] = None,
    need_kv: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Multi-head attention with RoPE, scalable softmax, and KV caching.

    A functional re-implementation of ``F.multi_head_attention_forward`` that
    accepts arbitrary leading batch dimensions and adds rotary embeddings,
    scalable softmax, and KV caching. Like the PyTorch original it takes raw
    weight *tensors* rather than modules, so the caller
    (:meth:`layers.MultiheadAttention.forward`) passes them in explicitly.

    ``query``, ``key`` and ``value`` are the three *input sequences*, each of
    width ``embed_dim``. They are not the Q/K/V of the attention formula; they
    are what ``in_proj_weight`` projects into them. Throughout this module the
    spelled-out names denote these pre-projection inputs and the lowercase
    ``q``/``k``/``v`` the post-projection, head-split tensors. Self-attention
    passes one tensor for all three; cross-attention (e.g. ``ISAB`` in
    ``layers.py``) passes different ones. See Notes for the full pipeline.

    Parameters
    ----------
    query : Tensor
        Input sequence projected into the queries, shape
        (..., tgt_len, embed_dim). Its length sets the output length.

    num_heads : int
        Number of attention heads. Must divide ``embed_dim``; each head then
        operates in ``head_dim = embed_dim // num_heads`` dimensions.

    in_proj_weight : Tensor
        The Q, K and V input projections packed into one matrix of shape
        (3 * embed_dim, embed_dim), stacked along the output axis as
        ``[W_q ; W_k ; W_v]``. Packing lets one GEMM compute all three when the
        input sequences coincide. Each (embed_dim, embed_dim) block is itself
        every head's projection stacked, which is why the head split is a
        reshape rather than a separate matmul. ``in_proj_weight[:embed_dim]``
        recovers ``W_q`` alone -- what the ``cached_kv`` path uses.

    in_proj_bias : Tensor
        Matching packed bias of shape (3 * embed_dim,).

    dropout_p : float
        Dropout probability applied to attention weights. Forced to 0.0 when
        ``training`` is False.

    out_proj_weight : Tensor
        Output projection of shape (embed_dim, embed_dim), applied after the
        heads are concatenated back to ``embed_dim``. This is the step that
        mixes information across heads.

    out_proj_bias : Tensor
        Output projection bias of shape (embed_dim,).

    key : Optional[Tensor], default=None
        Input sequence projected into the keys, shape
        (..., src_len, embed_dim). Required when ``cached_kv`` is None. Must
        match ``value`` in shape, since both index the same ``src_len`` source
        positions; ``src_len`` need not equal ``tgt_len``.

    value : Optional[Tensor], default=None
        Input sequence projected into the values, shape
        (..., src_len, embed_dim). Required when ``cached_kv`` is None.

    cached_kv : Optional[KVCacheEntry], default=None
        Pre-computed key and value projections for caching. When provided:

        - key and value parameters are ignored
        - Only query projection is computed
        - cached_kv.key shape: (..., num_heads, src_len, head_dim)
        - cached_kv.value shape: (..., num_heads, src_len, head_dim)
        - RoPE is applied only to queries (keys should already have RoPE applied)

    training : bool, default=True
        Whether the model is in training mode (affects dropout).

    key_padding_mask : Optional[Tensor], default=None
        Mask of shape (..., src_len) that identifies padding elements
        in the key sequence to be ignored:

        - For binary masks: True values indicate positions to ignore.
        - For float masks: Values are directly added to attention scores.

    attn_mask : Optional[Tensor], default=None
        Additive mask of shape (tgt_len, src_len) or
        (..., num_heads, tgt_len, src_len); the 2D form is broadcast to the
        latter. Expected to be a float mask holding ``-inf`` at blocked
        positions -- the caller converts boolean masks beforehand via
        ``F._canonical_mask``. Merged with ``key_padding_mask`` by addition.

    rope : Optional[RotaryEmbedding]
        Rotary position embedding, applied to ``q`` and ``k`` after the head
        split. Never applied to ``v``: RoPE encodes position through the
        query-key dot product, and ``v`` never enters one.

    ssmax_layer : Optional[nn.Module], default=None
        Scalable-softmax layer. Rescales ``q`` by a learned, sequence-length
        dependent factor before attention, which is equivalent to rescaling the
        logits and lets attention sharpness adapt to ``src_len``. SDPA's own
        1/sqrt(head_dim) scaling still applies on top. Applied inside
        :func:`sdpa_with_flattened_batch`, not here.

    need_kv : bool, default=False
        If True and ``cached_kv`` is None, also returns the computed K and V
        projections along with the attention output. Useful for caching K/V for
        subsequent calls.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor, Tensor]]
        If ``need_kv`` is False or ``cached_kv`` is provided:
            Attention output tensor of shape (..., tgt_len, embed_dim).
        If ``need_kv`` is True and ``cached_kv`` is None:
            Tuple of (attn_output, k, v) where:

            - attn_output: shape (..., tgt_len, embed_dim)
            - k: shape (..., num_heads, src_len, head_dim)
            - v: shape (..., num_heads, src_len, head_dim)

    Notes
    -----
    The pass proceeds in six steps:

    1. **In-projection.** ``query``/``key``/``value`` become ``q``/``k``/``v``,
       still (..., seq_len, embed_dim). ``F._in_projection_packed`` dispatches
       on object *identity*: one GEMM when ``query is key is value``, two when
       ``key is value`` (q separately, kv packed), three otherwise. Passing the
       same tensor object -- not an equal copy -- is what enables the fused
       path, so callers should preserve identity rather than recompute.
    2. **Head split.** Reshape to (..., num_heads, seq_len, head_dim). Nothing
       is computed here; the heads were already laid out side by side in the
       projection's output.
    3. **RoPE** on ``q`` and ``k`` only, once they are head-split.
    4. **Mask assembly.** ``attn_mask`` is validated and broadcast to
       (..., num_heads, tgt_len, src_len), then ``key_padding_mask`` is added
       into it. Dropout is disabled outside training.
    5. **Attention.** :func:`sdpa_with_flattened_batch` collapses the leading
       batch dims into one (required to trigger Flash Attention), applies SSMax
       to ``q``, then runs FlashAttention-3 or
       ``F.scaled_dot_product_attention``.
    6. **Head concat and out-projection** back to (..., tgt_len, embed_dim).

    When ``cached_kv`` is supplied, steps 1-3 reduce to projecting ``q`` alone
    through ``in_proj_weight[:embed_dim]``; ``k`` and ``v`` are read from the
    cache already head-split and already rotated.
    """

    # Extract shape information, supporting arbitrary batch dimensions
    *batch_shape, tgt_len, embed_dim = query.shape
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    if cached_kv is None:
        # Project the three input sequences into per-head q, k, v. Packed into
        # one GEMM when query/key/value are the same object (see Notes).
        if key is None or value is None:
            raise ValueError("key and value must be provided when cached_kv is None")
        src_len = key.shape[-2]
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
        k = k.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)
        v = v.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)
            k = rope.rotate_queries_or_keys(k)
    else:
        # Incremental path: k/v are already projected, head-split and rotated,
        # so only the query needs projecting -- hence the W_q row slice.
        k, v = cached_kv.key, cached_kv.value
        src_len = k.shape[-2]
        q_proj_weight = in_proj_weight[:embed_dim]
        q_proj_bias = in_proj_bias[:embed_dim] if in_proj_bias is not None else None
        q = F.linear(query, q_proj_weight, q_proj_bias)
        q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)

    # Disable dropout during evaluation
    if not training:
        dropout_p = 0.0

    # Process attention mask
    correct_2d_shape = (tgt_len, src_len)
    correct_nd_shape = (*batch_shape, num_heads, tgt_len, src_len)
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            if attn_mask.shape != correct_2d_shape:
                raise ValueError(f"2D attn_mask should have shape {correct_2d_shape}, but got {attn_mask.shape}")
            attn_mask = attn_mask.expand(*batch_shape, num_heads, tgt_len, src_len)
        elif attn_mask.dim() == len(correct_nd_shape):
            if attn_mask.shape != correct_nd_shape:
                raise ValueError(
                    f"{len(correct_nd_shape)}D attn_mask should have shape {correct_nd_shape}, "
                    f"but got {attn_mask.shape}"
                )
        else:
            raise ValueError(f"attn_mask must be 2D or {len(correct_nd_shape)}D, got {attn_mask.dim()}D")

    # Process key padding mask
    if key_padding_mask is not None:
        if key_padding_mask.shape != (*batch_shape, src_len):
            raise ValueError(
                f"key_padding_mask should have shape {(*batch_shape, src_len)}, but got {key_padding_mask.shape}"
            )
        key_padding_mask = key_padding_mask.view(*batch_shape, 1, 1, src_len).expand(
            *batch_shape, num_heads, tgt_len, src_len
        )

        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    attn_output = sdpa_with_flattened_batch(
        q, k, v, attn_mask, dropout_p, ssmax_layer=ssmax_layer
    )  # (..., nh, tgt_len, hs)

    # Concatenate the heads back into embed_dim, then mix across them
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # (batch_shape, tgt_len, E)

    if need_kv and cached_kv is None:
        return attn_output, k, v

    return attn_output
