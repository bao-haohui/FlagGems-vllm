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

"""Ascend KV-only RoPE/quantize/insert, preserving vLLM's NVIDIA cache ABI."""

import functools
import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _get_vector_core_count(device_index: int) -> int:
    """Query once per device; do not bake a particular 910B SKU into the grid."""
    import torch_npu
    import triton.runtime.driver as driver

    with torch_npu.npu.device(device_index):
        try:
            props = driver.active.utils.get_device_properties(device_index)
            count = int(props["num_vectorcore"])
        except (AttributeError, KeyError, RuntimeError):
            # Older FlagTree runtimes expose the same device capability
            # through torch_npu instead of the Triton property dictionary.
            count = int(torch.npu.get_device_limit(device_index)["vector_core_num"])
    if not 1 <= count <= 65535:
        raise RuntimeError(f"invalid Ascend vector-core count: {count}")
    return count


@triton.jit
def _f32_to_e4m3_bits(values):
    """Software E4M3FN RN-even with finite saturation; no FP8 typed pointer.

    Based on the existing Ascend Q+KV implementation, with saturation added
    because per-tensor scaling does not guarantee abs(values)<=448.
    """
    original_bits = values.to(tl.int32, bitcast=True)
    sign = (original_bits >> 24) & 0x80
    magnitude = tl.minimum(tl.abs(values), 448.0)
    bits = magnitude.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) - 120
    mantissa = (bits >> 20) & 7
    round_bit = (bits >> 19) & 1
    sticky = (bits & 0x7FFFF) != 0
    mantissa = mantissa + ((round_bit == 1) & (sticky | ((mantissa & 1) == 1))).to(
        tl.int32
    )
    exponent = exponent + (mantissa > 7).to(tl.int32)
    mantissa = tl.where(mantissa > 7, 0, mantissa)
    # FP8 subnormals are multiples of 2^-9. The FP32 magic addition rounds
    # to an integer with ties to even, including the subnormal/normal boundary.
    subnormal = ((magnitude * 512.0 + 8388608.0) - 8388608.0).to(tl.int32)
    encoded = tl.where(exponent >= 1, (exponent << 3) | mantissa, subnormal)
    encoded = tl.where(values != values, 0x7F, encoded)
    return (sign | encoded).to(tl.uint8)


@triton.jit
def _quantize_ue8m0(values):
    """Quantize each BF16-origin row; avoid scalar float-to-integer casts."""
    absmax = tl.maximum(tl.max(tl.abs(values), axis=1), 1e-4)
    raw_scale = absmax / 448.0
    bits = raw_scale.to(tl.int32, bitcast=True)
    code = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    # code=e+127, so the reciprocal's biased exponent is 127-e=254-code.
    inverse = ((254 - code) << 23).to(tl.float32, bitcast=True)
    quantized = _f32_to_e4m3_bits(values * inverse[:, None])
    return quantized, code.to(tl.uint8)


@triton.jit(
    do_not_specialize=["num_tokens", "cache_block_stride", "cache_token_stride"]
)
def _fused_deepseek_v4_kv_rope_insert_kernel(
    kv,
    cache_bytes,
    cache_bf16,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    fp8_scale,
    num_tokens,
    cache_block_stride,
    cache_token_stride,
    CACHE_BLOCK_SIZE: tl.constexpr,
    CACHE_KIND: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    groups = tl.arange(0, 512 // GROUP_SIZE)
    columns = tl.arange(0, GROUP_SIZE)
    dims = groups[:, None] * GROUP_SIZE + columns[None, :]
    is_rope = dims >= 448
    pair = (dims % 64) // 2

    # PR #849's core-bound loop avoids thousands of tiny program launches.
    # Both group/dimension offsets stay INT32; only scalar bases widen.
    for token in range(pid, num_tokens, NUM_PROGRAMS):
        slot = tl.load(slot_mapping + token)
        if slot >= 0:
            kv_base = kv + token.to(tl.int64) * 512
            position = tl.load(position_ids + token)
            cs_base = cos_sin_cache + position * 64
            values = tl.load(kv_base + dims).to(tl.float32)
            partner = tl.load(kv_base + (dims ^ 1), is_rope, other=0).to(tl.float32)
            cosine = tl.load(cs_base + pair, is_rope, other=0)
            sine = tl.load(cs_base + 32 + pair, is_rope, other=0)
            even = values * cosine - partner * sine
            odd = tl.fma(partner, sine, values * cosine)
            rotated = tl.where(is_rope, tl.where((dims & 1) == 0, even, odd), values)

            # Live slots fit INT32 by the host allocation check. Scalar page
            # byte offsets remain INT64, including padded/cross-layer strides.
            slot32 = slot.to(tl.int32)
            block = slot32 // CACHE_BLOCK_SIZE
            offset = (slot32 % CACHE_BLOCK_SIZE).to(tl.int64)
            page = block.to(tl.int64) * cache_block_stride

            if CACHE_KIND == 0:
                dst = (page + offset * cache_token_stride) // 2
                tl.store(cache_bf16 + dst + dims, rotated.to(tl.bfloat16))
            elif CACHE_KIND == 1:
                scaled = rotated * (1.0 / tl.load(fp8_scale))
                dst = page + offset * cache_token_stride
                tl.store(cache_bytes + dst + dims, _f32_to_e4m3_bits(scaled))
            elif CACHE_KIND == 2:
                quantized, codes = _quantize_ue8m0(values)
                dst = page + offset * 576
                tl.store(cache_bytes + dst + dims, quantized, dims < 448)
                # The wrapper passes a BF16 view, avoiding pointer-width casts.
                rope_dst = (dst + 448) // 2
                tl.store(
                    cache_bf16 + rope_dst + dims % 64,
                    rotated.to(tl.bfloat16),
                    is_rope,
                )
                scales = page + CACHE_BLOCK_SIZE * 576 + offset * 8
                tl.store(cache_bytes + scales + groups, codes, groups < 7)
                tl.store(cache_bytes + scales + 7, tl.full((), 0, tl.uint8))
            else:
                rounded = rotated.to(tl.bfloat16).to(tl.float32)
                quantized, codes = _quantize_ue8m0(rounded)
                dst = page + offset * 512
                tl.store(cache_bytes + dst + dims, quantized)
                scales = page + CACHE_BLOCK_SIZE * 512 + offset * 16
                tl.store(cache_bytes + scales + groups, codes)


def _cache_views(kv, k_cache, cache_kind):
    """No-copy views only; the compiler never sees an FP8 cache pointer."""
    cache_bytes = k_cache.view(torch.uint8)
    if cache_kind == 0:
        cache_bf16 = k_cache
    elif cache_kind == 2:
        if any(s % 2 for s in k_cache.stride()[:-1]) or k_cache.storage_offset() % 2:
            raise NotImplementedError(
                "packed V4 requires BF16-aligned page/token strides and storage offset"
            )
        cache_bf16 = k_cache.view(torch.bfloat16)
    else:
        # Unused constexpr branches are eliminated, so no cache conversion or
        # temporary allocation is needed for FP8 and packed MXFP8.
        cache_bf16 = kv
    return cache_bytes, cache_bf16


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
    """Rotate and optionally quantize KV, then update the paged SWA cache.

    BF16, per-tensor E4M3, packed V4 and packed MXFP8 retain the vLLM v0.32
    interface. Runtime core count controls the grid; semantic group sizes
    are fixed by cache format. No CUDA-specific launch knobs or Torch compute.
    """
    # Backend exports are imported while the package is initializing.
    from flaggems_vllm.ops.fused_deepseek_v4_kv_rope_insert import _validate_inputs

    logger.debug("GEMS_ASCEND FUSED_DEEPSEEK_V4_KV_ROPE_INSERT")
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
    if kv.device.type not in ("npu", "privateuseone"):
        raise RuntimeError("the Ascend override requires an NPU device")
    if k_cache.shape[0] * cache_block_size >= 2**31:
        raise NotImplementedError("Ascend cache allocations must have <2^31 slots")

    cache_bytes, cache_bf16 = _cache_views(kv, k_cache, cache_kind)
    with torch_device_fn.device(kv.device):
        programs = min(slot_mapping.numel(), _get_vector_core_count(kv.device.index))
        _fused_deepseek_v4_kv_rope_insert_kernel[(programs,)](
            kv,
            cache_bytes,
            cache_bf16,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            fp8_scale,
            slot_mapping.numel(),
            cache_bytes.stride(0),
            cache_bytes.stride(1),
            CACHE_BLOCK_SIZE=cache_block_size,
            CACHE_KIND=cache_kind,
            NUM_PROGRAMS=programs,
            GROUP_SIZE=32 if cache_kind == 3 else 64,
        )
