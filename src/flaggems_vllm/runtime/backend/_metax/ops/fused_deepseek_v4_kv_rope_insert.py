# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MetaX KV-only RoPE and paged cache insertion for DeepSeek V4."""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@triton.jit
def _multiply_preserve_zero(left, right):
    # Vector FMUL can discard -0 on this backend. FMA with a zero carrying
    # the product's sign has exactly FMUL's finite-input rounding and zero
    # sign, while retaining an explicit third operand in the generated code.
    sign = (left.to(tl.int32, bitcast=True) ^ right.to(tl.int32, bitcast=True)) & (
        -2147483648
    )
    return tl.fma(left, right, sign.to(tl.float32, bitcast=True))


@triton.jit
def _fp8_bytes(values, unscaled):
    # Preserve the sign before scaling: the MetaX multiply can canonicalize
    # an exact negative zero. All supported quantization scales are positive.
    sign = (unscaled.to(tl.int32, bitcast=True) >> 24) & 0x80
    encoded = tl.clamp(values, -448.0, 448.0).to(tl.float8e4nv)
    bits = encoded.to(tl.uint8, bitcast=True)
    return ((bits & 0x7F) | sign).to(tl.uint8)


@triton.jit
def _quantize_ue8m0(values, BLOCK_TOKENS: tl.constexpr, GROUP_SIZE: tl.constexpr):
    groups: tl.constexpr = 512 // GROUP_SIZE
    tiles = tl.reshape(values, (BLOCK_TOKENS, groups, GROUP_SIZE))
    absmax = tl.maximum(tl.max(tl.abs(tiles), axis=2), 1e-4)
    # For finite BF16-origin values, scale is normal FP32. ceil(log2(scale))
    # is its biased exponent plus one iff the mantissa is nonzero. Construct
    # the reciprocal power of two exactly, avoiding two transcendental ops.
    bits = (absmax / 448.0).to(tl.int32, bitcast=True)
    code = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    inverse = ((254 - code) << 23).to(tl.float32, bitcast=True)
    scaled = tiles * inverse[:, :, None]
    quantized = _fp8_bytes(scaled, tiles)
    return tl.reshape(quantized, (BLOCK_TOKENS, 512)), code.to(tl.uint8)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fused_deepseek_v4_kv_rope_insert"),
    key=[
        "num_tokens",
        "CACHE_KIND",
        "CACHE_BLOCK_SIZE",
        "cache_block_stride",
        "cache_token_stride",
    ],
    use_cuda_graph=True,
)
@triton.jit
def _fused_deepseek_v4_kv_rope_insert_kernel(
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    fp8_scale,
    num_tokens,
    cache_block_stride,
    cache_token_stride,
    CACHE_BLOCK_SIZE: tl.constexpr,
    CACHE_KIND: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    slots = tl.load(slot_mapping + tokens, tokens < num_tokens, other=-1)
    live = (tokens < num_tokens) & (slots >= 0)
    # Mask before following positions: skipped rows may have invalid position
    # IDs, including an entirely empty cos/sin table when all slots are negative.
    positions = tl.load(position_ids + tokens, live, other=0)
    pairs = tl.arange(0, 32)
    cs = cos_sin_cache + positions[:, None] * 64 + pairs[None, :]
    cosine = tl.load(cs, live[:, None], other=0.0)
    sine = tl.load(cs + 32, live[:, None], other=0.0)

    dims = tl.arange(0, 512)
    kv_base = kv + tokens[:, None].to(tl.int64) * 512
    if CACHE_KIND == 2:
        values = tl.load(
            kv_base + dims[None, :], live[:, None] & (dims[None, :] < 448), other=0
        ).to(tl.float32)
    else:
        values = tl.load(kv_base + dims[None, :], live[:, None], other=0).to(tl.float32)
    rope_dims = tl.arange(0, 64)
    rope = tl.load(kv_base + 448 + rope_dims[None, :], live[:, None], other=0).to(
        tl.float32
    )
    even, odd = tl.split(tl.reshape(rope, (BLOCK_TOKENS, 32, 2)))
    # The official MACA build contracts the cos product in both components.
    # Express both FMAs explicitly: implicit contraction is compiler-dependent,
    # and PR #1083's odd-component even*sin FMA differs at rounding boundaries.
    odd_sine = _multiply_preserve_zero(odd, sine)
    # Vector floating negation can also canonicalize zero; flip the IEEE sign.
    negative_odd_sine = (odd_sine.to(tl.int32, bitcast=True) ^ -2147483648).to(
        tl.float32, bitcast=True
    )
    rotated_rope = tl.reshape(
        tl.join(
            tl.fma(even, cosine, negative_odd_sine),
            tl.fma(odd, cosine, _multiply_preserve_zero(even, sine)),
        ),
        (BLOCK_TOKENS, 64),
    )

    safe_slots = tl.maximum(slots, 0)
    blocks = safe_slots // CACHE_BLOCK_SIZE
    offsets = safe_slots % CACHE_BLOCK_SIZE
    page = k_cache + blocks[:, None] * cache_block_stride

    if CACHE_KIND == 2:
        payload, codes = _quantize_ue8m0(values, BLOCK_TOKENS, 64)
        token = page + offsets[:, None] * 576
        tl.store(
            token + dims[None, :],
            payload,
            live[:, None] & (dims[None, :] < 448),
        )
        rope_dst = (token + 448).to(tl.pointer_type(tl.bfloat16))
        tl.store(
            rope_dst + rope_dims[None, :], rotated_rope.to(tl.bfloat16), live[:, None]
        )
        scale_dims = tl.arange(0, 8)
        scale = page + CACHE_BLOCK_SIZE * 576 + offsets[:, None] * 8
        # Write all eight bytes, including the mandatory zero padding byte.
        tl.store(
            scale + scale_dims[None, :],
            tl.where(scale_dims[None, :] < 7, codes, 0).to(tl.uint8),
            live[:, None],
        )
    else:
        rows = tl.reshape(values, (BLOCK_TOKENS, 8, 64))
        rotated = tl.reshape(
            tl.where(
                tl.arange(0, 8)[None, :, None] == 7,
                rotated_rope[:, None, :],
                rows,
            ),
            (BLOCK_TOKENS, 512),
        )
        if CACHE_KIND == 0:
            dst = page + offsets[:, None] * cache_token_stride + dims[None, :]
            tl.store(dst, rotated.to(tl.bfloat16), live[:, None])
        elif CACHE_KIND == 1:
            dst = page + offsets[:, None] * cache_token_stride + dims[None, :]
            # Per-tensor FP8 quantizes FP32 RoPE directly: no BF16 round trip.
            scaled = rotated * (1.0 / tl.load(fp8_scale))
            tl.store(
                dst.to(tl.pointer_type(tl.uint8)),
                _fp8_bytes(scaled, rotated),
                live[:, None],
            )
        else:
            # MXFP8, unlike per-tensor FP8, rounds RoPE to BF16 first.
            rounded = rotated.to(tl.bfloat16).to(tl.float32)
            payload, codes = _quantize_ue8m0(rounded, BLOCK_TOKENS, 32)
            dst = page + offsets[:, None] * 512 + dims[None, :]
            tl.store(dst, payload, live[:, None])
            scale_dims = tl.arange(0, 16)
            scale = page + CACHE_BLOCK_SIZE * 512 + offsets[:, None] * 16
            tl.store(scale + scale_dims[None, :], codes, live[:, None])


def fused_deepseek_v4_kv_rope_insert(
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache_block_size: int,
    fp8_scale: Optional[torch.Tensor] = None,
    kv_mxfp8: bool = False,
) -> None:
    """Apply KV RoPE and insert BF16/FP8/V4/MXFP8 rows into the SWA cache.

    The MetaX backend tunes the token tile and warp count together, instead
    of assuming NVIDIA's one-warp/token launch maps to the 64-lane device.
    Live slots must be unique and in range; their positions must be valid.
    """
    # Backend exports are imported during package initialization.
    from flaggems_vllm.ops.fused_deepseek_v4_kv_rope_insert import _validate_inputs

    logger.debug("GEMS_METAX FUSED_DEEPSEEK_V4_KV_ROPE_INSERT")
    cache_kind = _validate_inputs(
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        cache_block_size,
        fp8_scale,
        kv_mxfp8,
    )
    if cache_kind is None:
        return
    if kv.device.type != "cuda":
        raise RuntimeError("the MetaX override requires a CUDA-compatible device")
    if cache_kind >= 2:
        if k_cache.stride(1) != k_cache.shape[2]:
            raise NotImplementedError("packed cache requires a contiguous page layout")
        page_elements = cache_block_size * k_cache.shape[2]
        if cache_kind == 2 and (k_cache.stride(0) % 2 or k_cache.storage_offset() % 2):
            raise NotImplementedError("packed V4 requires BF16-aligned pages")
    else:
        if k_cache.stride(1) < 512:
            raise NotImplementedError("cache token records must not overlap")
        page_elements = (cache_block_size - 1) * k_cache.stride(1) + 512
    if k_cache.stride(0) < page_elements:
        raise NotImplementedError("cache pages must not overlap")

    with torch_device_fn.device(kv.device):
        _fused_deepseek_v4_kv_rope_insert_kernel[
            lambda meta: (triton.cdiv(slot_mapping.numel(), meta["BLOCK_TOKENS"]),)
        ](
            kv,
            k_cache,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            fp8_scale,
            slot_mapping.numel(),
            k_cache.stride(0),
            k_cache.stride(1),
            CACHE_BLOCK_SIZE=cache_block_size,
            CACHE_KIND=cache_kind,
            enable_fp_fusion=False,
        )
