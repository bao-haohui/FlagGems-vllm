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

"""Native collection contract and independent Torch reference; never production."""

import importlib

import torch

OP_NAME = "fused_deepseek_v4_kv_rope_insert"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

# Input KV is always BF16; cache format is an independent collection dimension.
CACHE_FORMATS = ("bf16", "fp8", "packed_v4", "mxfp8")

# FlagOSTune covers concurrency 1/2/4/8/64/100, up to 2048 sequences and
# 16384 batched tokens. vLLM uses 32-token pages for this SWA cache.
BENCHMARK_SHAPES = [
    (1, 1, 32),
    (2, 2, 32),
    (4, 4, 32),
    (8, 8, 32),
    (64, 64, 32),
    (100, 100, 32),
    (1027, 1024, 32),
    (2048, 2048, 32),
    (8192, 8192, 32),
    (16384, 16384, 32),
]


def native(device=None):
    if device is not None and torch.device(device).type != "cuda":
        return None
    try:
        importlib.import_module("vllm._custom_ops")
    except (ImportError, AttributeError, RuntimeError, OSError):
        return None
    op = getattr(torch.ops._C, OP_NAME, None)
    return op if callable(op) else None


def _integer_power_of_two(exponent):
    """Exact normal FP32 powers for the finite FP8 quantization exponent range.

    NPU exp2 can differ by one ULP even for integral exponents. That shifts
    exact E4M3 midpoints, so construct the IEEE exponent instead of using exp2.
    """
    return ((exponent.to(torch.int32) + 127) << 23).view(torch.float32)


def encode_e4m3(values):
    """Independent arithmetic E4M3 encoder; no device FP8 cast is needed.

    Quantize to the exponent-dependent lattice with torch.round (ties to
    even), rather than reusing the Triton encoder's FP32 mantissa bit logic.
    """
    values = values.float()
    magnitude = values.abs().clamp(max=448.0)
    exponent = torch.floor(torch.log2(magnitude.clamp_min(2.0**-6)))
    inverse_step = _integer_power_of_two(3 - exponent)
    rounded = torch.round(magnitude * inverse_step).to(torch.int32)
    normal = (exponent.to(torch.int32) + 7) * 8 + rounded - 8
    encoded = torch.where(magnitude >= 2.0**-6, normal, rounded)
    encoded = torch.where(torch.isnan(values), 0x7F, encoded)
    # Some NPU signbit implementations drop the sign of -0.0. Read the IEEE
    # sign directly while keeping the independent arithmetic rounding above.
    sign = (values.view(torch.int32) >> 24) & 0x80
    return (sign | encoded).to(torch.uint8)


def _quantize_reference(values, group_size):
    groups = values.reshape(values.shape[0], -1, group_size)
    absmax = groups.abs().amax(dim=-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(absmax / 448.0))
    normalized = groups * _integer_power_of_two(-exponent).unsqueeze(-1)
    return encode_e4m3(normalized).reshape(values.shape), (exponent + 127).to(
        torch.uint8
    )


def torch_reference(
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    cache_block_size,
    fp8_scale=None,
    kv_mxfp8=False,
    *,
    _assume_all_valid=False,
):
    """Complete unfused reference, preserving input KV and physical cache gaps.

    _assume_all_valid is a benchmark-only contract: its input builder supplies
    exclusively live slots, avoiding data-dependent nonzero during graph capture.
    General correctness tests retain negative-slot coverage.
    """
    count = slot_mapping.numel()
    if count == 0:
        return
    if _assume_all_valid:
        rows = torch.arange(count, device=kv.device)
        slots = slot_mapping
    else:
        rows = torch.nonzero(slot_mapping >= 0, as_tuple=True)[0]
        slots = slot_mapping[rows]
        if rows.numel() == 0:
            return
    values = kv[rows].float()
    angles = cos_sin_cache[position_ids[rows]]
    cosine, sine = angles[:, :32], angles[:, 32:]
    pairs = values[:, 448:].reshape(-1, 32, 2)
    rotated = values.clone()
    rotated[:, 448::2] = pairs[:, :, 0] * cosine - pairs[:, :, 1] * sine
    rotated[:, 449::2] = torch.addcmul(pairs[:, :, 1] * cosine, pairs[:, :, 0], sine)
    block = slots // cache_block_size
    offset = slots % cache_block_size

    if k_cache.dtype == torch.bfloat16:
        k_cache[block, offset] = rotated.to(torch.bfloat16)
    elif k_cache.dtype == torch.float8_e4m3fn:
        # Reinterpret storage only: even an FP8-incomplete device can compare
        # the resulting E4M3 bytes without executing an FP8 cast kernel.
        k_cache.view(torch.uint8)[block, offset] = encode_e4m3(
            rotated * (1.0 / fp8_scale)
        )
    else:
        width = k_cache.shape[2]
        pages = k_cache.as_strided(
            (k_cache.shape[0], cache_block_size * width),
            (k_cache.stride(0), 1),
        )
        rounded = rotated.to(torch.bfloat16)
        if kv_mxfp8:
            data_bytes, scale_bytes = 512, 16
            payload, scales = _quantize_reference(rounded.float(), 32)
        else:
            data_bytes, scale_bytes = 576, 8
            quantized, codes = _quantize_reference(rounded[:, :448].float(), 64)
            payload = torch.cat(
                (quantized, rounded[:, 448:].contiguous().view(torch.uint8)),
                dim=-1,
            )
            scales = torch.cat((codes, torch.zeros_like(codes[:, :1])), dim=-1)
        data_end = cache_block_size * data_bytes
        data_view = pages[:, :data_end].view(-1, cache_block_size, data_bytes)
        scale_view = pages[:, data_end:].view(-1, cache_block_size, scale_bytes)
        data_view[block, offset] = payload
        scale_view[block, offset] = scales


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "num_tokens_insert": num_insert,
            "cache_block_size": block_size,
            "cache_format": cache_format,
        }
        for num_tokens, num_insert, block_size in BENCHMARK_SHAPES
        for cache_format in CACHE_FORMATS
    ]


def _cache_spec(cache_format):
    return {
        "bf16": (torch.bfloat16, 512),
        "fp8": (torch.float8_e4m3fn, 512),
        "packed_v4": (torch.uint8, 584),
        "mxfp8": (torch.uint8, 528),
    }[cache_format]


def build_inputs(
    binding,
    dtype,
    device,
    *,
    padded_pages=False,
    slot_mapping=None,
    position_offset=0,
    scale_value=0.5,
):
    num_tokens = binding["num_tokens"]
    num_insert = binding["num_tokens_insert"]
    block_size = binding["cache_block_size"]
    cache_format = binding["cache_format"]
    cache_dtype, width = _cache_spec(cache_format)
    kv = torch.randn(num_tokens, 512, dtype=dtype, device=device)
    num_blocks = (num_insert + block_size - 1) // block_size + 1
    page_stride = block_size * width
    if padded_pages:
        page_stride = ((page_stride + 575) // 576) * 576 + 576
    itemsize = 2 if cache_dtype == torch.bfloat16 else 1
    # Raw bytes, as in PR #849: creating test inputs must not need an FP8 cast.
    backing = torch.full(
        (num_blocks, page_stride * itemsize), 0xA5, dtype=torch.uint8, device=device
    ).view(cache_dtype)
    cache = backing.as_strided((num_blocks, block_size, width), (page_stride, width, 1))
    # Unique shuffled slots stress paged writes; larger cases mask one row.
    if slot_mapping is None:
        slots = torch.randperm(num_insert, dtype=torch.int64, device=device)
        if num_insert > 1:
            slots[-1] = -1
    else:
        slots = slot_mapping
    positions = torch.arange(
        position_offset,
        position_offset + num_tokens,
        dtype=torch.int64,
        device=device,
    )
    angle_positions = torch.arange(
        position_offset + num_tokens, dtype=torch.int64, device=device
    )
    angles = (
        angle_positions.float()[:, None]
        * (10000.0 ** (-torch.arange(0, 64, 2, device=device).float() / 64))[None, :]
    )
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    scale = (
        torch.tensor([scale_value], dtype=torch.float32, device=device)
        if cache_format == "fp8"
        else None
    )
    return (
        kv,
        cache,
        slots,
        positions,
        cos_sin,
        block_size,
        scale,
        cache_format == "mxfp8",
    ), {}


def key_shape(binding):
    return (
        f"tokens{binding['num_tokens']}_insert{binding['num_tokens_insert']}_"
        f"bs{binding['cache_block_size']}_{binding['cache_format']}"
    )


def config(binding, dtype):
    num_tokens = binding["num_tokens"]
    num_insert = binding["num_tokens_insert"]
    block_size = binding["cache_block_size"]
    cache_format = binding["cache_format"]
    cache_dtype, width = _cache_spec(cache_format)
    num_blocks = (num_insert + block_size - 1) // block_size + 1
    cache = {
        "shape": [num_blocks, block_size, width],
        "dtype": str(cache_dtype),
    }
    return {
        "inputs": {
            "kv": {"shape": [num_tokens, 512], "dtype": str(dtype)},
            "k_cache": dict(cache),
            "slot_mapping": {"shape": [num_insert], "dtype": "torch.int64"},
            "position_ids": {"shape": [num_tokens], "dtype": "torch.int64"},
            "cos_sin_cache": {"shape": [num_tokens, 64], "dtype": "torch.float32"},
            "cache_block_size": {"scalar": block_size},
            "fp8_scale": (
                {"shape": [1], "dtype": "torch.float32", "value": 0.5}
                if cache_format == "fp8"
                else {"scalar": None}
            ),
            "kv_mxfp8": {"scalar": cache_format == "mxfp8"},
        },
        "outputs": {"k_cache": dict(cache)},
        "dims": dict(binding),
    }
