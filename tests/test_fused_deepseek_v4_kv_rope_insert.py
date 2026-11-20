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

import functools

import pytest
import torch

import flaggems_vllm
from baseline.fused_deepseek_v4_kv_rope_insert import (
    CACHE_FORMATS,
    build_inputs,
    encode_e4m3,
    mthreads_cpu_reference,
    native,
    torch_reference,
)
from flaggems_vllm import testing as utils

from . import conftest as cfg

VLLM_REF = native(flaggems_vllm.device)
TORCH_REF = functools.partial(
    torch_reference, _rope_fma_on_cos=flaggems_vllm.vendor_name in ("metax", "thead")
)
if flaggems_vllm.vendor_name == "mthreads":
    TORCH_REF = mthreads_cpu_reference
REFERENCE = VLLM_REF or TORCH_REF

# Functional coverage retains DP padding and upstream's 256-token page cases.
TEST_SHAPES = [
    (1, 1, 32),
    (4, 1, 32),
    (20, 17, 32),
    (128, 117, 32),
    (1027, 1024, 32),
    (8192, 8192, 32),
    (4, 1, 256),
    (20, 17, 256),
    (1026, 1023, 256),
    (1027, 1024, 256),
]

SHAPES = (
    [(1, 1, 32), (20, 17, 32), (20, 17, 256), (1026, 1023, 256)]
    if cfg.QUICK_MODE
    else TEST_SHAPES
)


def _make_inputs(
    num_tokens, num_tokens_insert, cache_block_size, cache_format, **input_kwargs
):
    binding = {
        "num_tokens": num_tokens,
        "num_tokens_insert": num_tokens_insert,
        "cache_block_size": cache_block_size,
        "cache_format": cache_format,
    }
    args, _ = build_inputs(
        binding, torch.bfloat16, flaggems_vllm.device, **input_kwargs
    )
    (
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        cache_block_size,
        fp8_scale,
        kv_mxfp8,
    ) = args
    return {
        "kv": kv,
        "k_cache": k_cache,
        "slot_mapping": slot_mapping,
        "position_ids": position_ids,
        "cos_sin_cache": cos_sin_cache,
        "cache_block_size": cache_block_size,
        "fp8_scale": fp8_scale,
        "kv_mxfp8": kv_mxfp8,
    }


def _assert_bytes_equal(actual, expected):
    utils.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), torch.uint8, atol=0
    )


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@pytest.mark.parametrize("num_tokens,num_tokens_insert,cache_block_size", SHAPES)
@torch.inference_mode()
def test_matches_reference(
    num_tokens, num_tokens_insert, cache_block_size, cache_format
):
    inputs = _make_inputs(num_tokens, num_tokens_insert, cache_block_size, cache_format)
    reference_cache = inputs["k_cache"].clone()
    kv_before = inputs["kv"].clone()

    REFERENCE(
        kv=inputs["kv"],
        k_cache=reference_cache,
        slot_mapping=inputs["slot_mapping"],
        position_ids=inputs["position_ids"],
        cos_sin_cache=inputs["cos_sin_cache"],
        cache_block_size=inputs["cache_block_size"],
        fp8_scale=inputs["fp8_scale"],
        kv_mxfp8=inputs["kv_mxfp8"],
    )
    result = flaggems_vllm.fused_deepseek_v4_kv_rope_insert(
        kv=inputs["kv"],
        k_cache=inputs["k_cache"],
        slot_mapping=inputs["slot_mapping"],
        position_ids=inputs["position_ids"],
        cos_sin_cache=inputs["cos_sin_cache"],
        cache_block_size=inputs["cache_block_size"],
        fp8_scale=inputs["fp8_scale"],
        kv_mxfp8=inputs["kv_mxfp8"],
    )

    assert result is None
    _assert_bytes_equal(inputs["kv"], kv_before)
    _assert_bytes_equal(inputs["k_cache"], reference_cache)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@torch.inference_mode()
def test_nontrivial_positions_slots_and_scale(cache_format):
    slots = torch.tensor(
        [9, 3, 11, -7, 0], dtype=torch.int64, device=flaggems_vllm.device
    )
    inputs = _make_inputs(
        8,
        5,
        4,
        cache_format,
        slot_mapping=slots,
        position_offset=7,
        scale_value=0.7,
    )
    reference_cache = inputs["k_cache"].clone()

    REFERENCE(
        kv=inputs["kv"],
        k_cache=reference_cache,
        slot_mapping=inputs["slot_mapping"],
        position_ids=inputs["position_ids"],
        cos_sin_cache=inputs["cos_sin_cache"],
        cache_block_size=inputs["cache_block_size"],
        fp8_scale=inputs["fp8_scale"],
        kv_mxfp8=inputs["kv_mxfp8"],
    )
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(
        kv=inputs["kv"],
        k_cache=inputs["k_cache"],
        slot_mapping=inputs["slot_mapping"],
        position_ids=inputs["position_ids"],
        cos_sin_cache=inputs["cos_sin_cache"],
        cache_block_size=inputs["cache_block_size"],
        fp8_scale=inputs["fp8_scale"],
        kv_mxfp8=inputs["kv_mxfp8"],
    )

    _assert_bytes_equal(inputs["k_cache"], reference_cache)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@torch.inference_mode()
def test_matches_reference_with_padded_pages(cache_format):
    cache_block_size = 256
    num_insert = 17
    num_blocks = (num_insert + cache_block_size - 1) // cache_block_size + 1
    slots = torch.randperm(num_blocks * cache_block_size, device=flaggems_vllm.device)[
        :num_insert
    ]
    slots[::5] = -1
    inputs = _make_inputs(
        20,
        num_insert,
        cache_block_size,
        cache_format,
        padded_pages=True,
        slot_mapping=slots,
    )
    page_stride = inputs["k_cache"].stride(0)
    backing = inputs["k_cache"].as_strided((num_blocks, page_stride), (page_stride, 1))
    reference_backing = backing.clone()
    reference_cache = reference_backing.as_strided(
        inputs["k_cache"].shape, inputs["k_cache"].stride()
    )
    kv_before = inputs["kv"].clone()

    REFERENCE(
        kv=inputs["kv"],
        k_cache=reference_cache,
        slot_mapping=inputs["slot_mapping"],
        position_ids=inputs["position_ids"],
        cos_sin_cache=inputs["cos_sin_cache"],
        cache_block_size=inputs["cache_block_size"],
        fp8_scale=inputs["fp8_scale"],
        kv_mxfp8=inputs["kv_mxfp8"],
    )
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(
        kv=inputs["kv"],
        k_cache=inputs["k_cache"],
        slot_mapping=inputs["slot_mapping"],
        position_ids=inputs["position_ids"],
        cos_sin_cache=inputs["cos_sin_cache"],
        cache_block_size=inputs["cache_block_size"],
        fp8_scale=inputs["fp8_scale"],
        kv_mxfp8=inputs["kv_mxfp8"],
    )

    _assert_bytes_equal(backing, reference_backing)
    _assert_bytes_equal(inputs["kv"], kv_before)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@torch.inference_mode()
def test_empty_insert_is_noop(cache_format):
    inputs = _make_inputs(4, 0, 32, cache_format)
    cache_before = inputs["k_cache"].clone()
    kv_before = inputs["kv"].clone()

    result = flaggems_vllm.fused_deepseek_v4_kv_rope_insert(
        kv=inputs["kv"],
        k_cache=inputs["k_cache"],
        slot_mapping=inputs["slot_mapping"],
        position_ids=inputs["position_ids"],
        cos_sin_cache=inputs["cos_sin_cache"],
        cache_block_size=inputs["cache_block_size"],
        fp8_scale=inputs["fp8_scale"],
        kv_mxfp8=inputs["kv_mxfp8"],
    )

    assert result is None
    _assert_bytes_equal(inputs["kv"], kv_before)
    _assert_bytes_equal(inputs["k_cache"], cache_before)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
def test_all_negative_slots_are_noop(cache_format):
    slots = torch.tensor([-1, -7, -99], device=flaggems_vllm.device)
    inputs = _make_inputs(5, 3, 32, cache_format, slot_mapping=slots)
    before = inputs["k_cache"].view(torch.uint8).clone()
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)
    _assert_bytes_equal(inputs["k_cache"].view(torch.uint8), before)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("cache_format", ["bf16", "fp8"])
def test_noncontiguous_token_stride(cache_format):
    inputs = _make_inputs(20, 17, 32, cache_format)
    cache = inputs["k_cache"]
    blocks, block_size, width = cache.shape
    token_stride = width + 16
    block_stride = block_size * token_stride + 128
    itemsize = cache.element_size()
    backing = torch.full(
        (blocks, block_stride * itemsize),
        0xA5,
        dtype=torch.uint8,
        device=cache.device,
    )
    reference_backing = backing.clone()
    inputs["k_cache"] = backing.view(cache.dtype).as_strided(
        cache.shape, (block_stride, token_stride, 1)
    )
    reference_cache = reference_backing.view(cache.dtype).as_strided(
        cache.shape, (block_stride, token_stride, 1)
    )
    reference_args = dict(inputs, k_cache=reference_cache)
    REFERENCE(**reference_args)
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)
    _assert_bytes_equal(backing, reference_backing)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
def test_public_api_selects_backend_override():
    vendor = flaggems_vllm.vendor_name
    if vendor not in ("ascend", "metax", "thead", "mthreads"):
        pytest.skip("Backend override dispatch check")
    assert f"._{vendor}.ops." in (
        flaggems_vllm.fused_deepseek_v4_kv_rope_insert.__module__
    )


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@pytest.mark.parametrize("all_skipped", [False, True])
def test_skipped_positions_are_not_read(cache_format, all_skipped):
    if flaggems_vllm.vendor_name not in ("ascend", "metax", "thead", "mthreads"):
        pytest.skip("Backend extensions allow invalid positions on skipped rows")
    slots = torch.tensor([3, -7, 1, -99, 8], device=flaggems_vllm.device)
    if all_skipped:
        slots.fill_(-7)
    inputs = _make_inputs(8, 5, 32, cache_format, slot_mapping=slots)
    inputs["position_ids"][:5][slots < 0] = 2**40
    inputs["position_ids"][5:] = 2**40
    if all_skipped:
        inputs["cos_sin_cache"] = inputs["cos_sin_cache"][:0]
    expected = inputs["k_cache"].clone()
    # The raw native baseline reads positions before checking negative slots.
    # Use the independent reference for this additional input-safety contract.
    TORCH_REF(**dict(inputs, k_cache=expected))
    before = inputs["kv"].clone()
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)
    _assert_bytes_equal(inputs["k_cache"], expected)
    _assert_bytes_equal(inputs["kv"], before)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
def test_fp8_rounding_and_scale_boundaries(cache_format):
    # BF16 neighbors of every E4M3 midpoint, signed zero, saturation, tiny
    # groups and maxima exactly at/on either side of UE8M0 power boundaries.
    lattice = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = ((lattice[:-1] + lattice[1:]) * 0.5).bfloat16()
    below = torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf")))
    above = torch.nextafter(midpoints, torch.full_like(midpoints, float("inf")))
    positive = torch.cat((lattice.bfloat16(), midpoints, below, above))
    pattern = torch.cat((positive, -positive))
    inputs = _make_inputs(20, 17, 32, cache_format, scale_value=1.0)
    values = pattern.repeat((20 * 512 + pattern.numel() - 1) // pattern.numel())
    values = values[: 20 * 512].reshape(20, 512)
    # Set entire rows to known extrema so each scale group exercises the edge.
    values[0].zero_()
    values[1].fill_(2**-20)
    values[2].fill_(448.0)
    values[3].fill_(446.0)
    values[4].fill_(450.0)
    values[5].fill_(-448.0)
    values[6].fill_(57344.0)
    inputs["kv"] = values.to(flaggems_vllm.device)
    inputs["cos_sin_cache"][:, :32].fill_(1)
    inputs["cos_sin_cache"][:, 32:].zero_()
    cpu_inputs = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    TORCH_REF(**cpu_inputs)
    if VLLM_REF is not None:
        native_cache = inputs["k_cache"].clone()
        VLLM_REF(**dict(inputs, k_cache=native_cache))
        _assert_bytes_equal(native_cache.cpu(), cpu_inputs["k_cache"])
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)
    _assert_bytes_equal(inputs["k_cache"].cpu(), cpu_inputs["k_cache"])


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
def test_nonzero_cache_storage_offset(cache_format):
    inputs = _make_inputs(20, 17, 32, cache_format)
    cache = inputs["k_cache"]
    # Keep a prefix and suffix around the physical cache, both checked below.
    prefix_bytes = 32
    byte_count = cache.numel() * cache.element_size()
    storage = torch.full(
        (byte_count + prefix_bytes * 2,),
        0xA5,
        dtype=torch.uint8,
        device=cache.device,
    )
    expected = storage.clone()
    inputs["k_cache"] = (
        storage[prefix_bytes : prefix_bytes + byte_count]
        .view(cache.dtype)
        .view(cache.shape)
    )
    expected_cache = (
        expected[prefix_bytes : prefix_bytes + byte_count]
        .view(cache.dtype)
        .view(cache.shape)
    )
    REFERENCE(**dict(inputs, k_cache=expected_cache))
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)
    _assert_bytes_equal(storage, expected)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
def test_per_tensor_fp8_does_not_round_rope_through_bf16():
    inputs = _make_inputs(1, 1, 32, "fp8", scale_value=1.0)
    inputs["kv"].zero_()
    inputs["kv"][:, 448::2] = 1
    # The even component lies just above the FP8 midpoint 0.53125. An
    # incorrect BF16 intermediate rounds it back onto that midpoint, changing
    # the final E4M3 byte from 0.5625 to 0.5.
    cosine = 0.5315
    inputs["cos_sin_cache"][:, :32].fill_(cosine)
    inputs["cos_sin_cache"][:, 32:].fill_((1.0 - cosine * cosine) ** 0.5)
    cpu_inputs = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    TORCH_REF(**cpu_inputs)
    expected = cpu_inputs["k_cache"]
    assert expected[0, 0, 448].float().item() == 0.5625
    if VLLM_REF is not None:
        native_cache = inputs["k_cache"].clone()
        VLLM_REF(**dict(inputs, k_cache=native_cache))
        _assert_bytes_equal(native_cache.cpu(), expected)
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)
    _assert_bytes_equal(inputs["k_cache"].cpu(), expected)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@pytest.mark.parametrize(
    "even,odd,cosine,sine,dimension,expected_value,scale",
    [
        (-0.5, 2.015625, 0.5443320274353027, 0.8388698697090149, 449, 0.67578125, 1.0),
        (
            0.671875,
            -1.359375,
            0.9095131158828735,
            -0.4156752824783325,
            448,
            0.046142578125,
            0.04331343248486519,
        ),
    ],
)
def test_native_rope_fma_order(
    cache_format, even, odd, cosine, sine, dimension, expected_value, scale
):
    if flaggems_vllm.vendor_name not in ("metax", "thead"):
        pytest.skip("Vendor native cosine FMA contraction order")
    inputs = _make_inputs(1, 1, 32, cache_format, scale_value=scale)
    inputs["kv"].zero_()
    inputs["kv"][:, 448::2] = even
    inputs["kv"][:, 449::2] = odd
    inputs["cos_sin_cache"][:, :32].fill_(cosine)
    inputs["cos_sin_cache"][:, 32:].fill_(sine)
    cpu_inputs = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    TORCH_REF(**cpu_inputs)
    expected = cpu_inputs["k_cache"]
    if cache_format == "bf16":
        assert expected[0, 0, dimension].float().item() == expected_value
    if VLLM_REF is not None:
        native_cache = inputs["k_cache"].clone()
        VLLM_REF(**dict(inputs, k_cache=native_cache))
        _assert_bytes_equal(native_cache.cpu(), expected)
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)
    _assert_bytes_equal(inputs["k_cache"].cpu(), expected)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@pytest.mark.parametrize(
    "cosine,sine", [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
)
def test_native_rope_signed_zeros(cache_format, cosine, sine):
    if flaggems_vllm.vendor_name not in ("metax", "thead", "mthreads"):
        pytest.skip("Native RoPE signed-zero regression")
    slots = torch.arange(17, device=flaggems_vllm.device)
    inputs = _make_inputs(20, 17, 32, cache_format, scale_value=1.0, slot_mapping=slots)
    values = (
        torch.tensor([-0.0, 0.0], dtype=torch.bfloat16).repeat(20 * 256).view(20, 512)
    )
    values[:, 448:] = torch.tensor(
        [-0.0, 1.0, 0.0, -1.0, -1.0, -0.0, 1.0, 0.0], dtype=torch.bfloat16
    ).repeat(8)
    inputs["kv"] = values.to(flaggems_vllm.device)
    inputs["cos_sin_cache"][:, :32].fill_(cosine)
    inputs["cos_sin_cache"][:, 32:].fill_(sine)
    cpu_inputs = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    TORCH_REF(**cpu_inputs)
    expected = cpu_inputs["k_cache"]
    if VLLM_REF is not None:
        native_cache = inputs["k_cache"].clone()
        VLLM_REF(**dict(inputs, k_cache=native_cache))
        _assert_bytes_equal(native_cache.cpu(), expected)
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)
    _assert_bytes_equal(inputs["k_cache"].cpu(), expected)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("invalid", ["token_stride", "page_stride", "alignment"])
def test_backend_rejects_unsupported_packed_layout(invalid):
    if flaggems_vllm.vendor_name not in ("metax", "thead", "mthreads"):
        pytest.skip("Backend packed-layout validation")
    inputs = _make_inputs(4, 4, 32, "packed_v4")
    shape = inputs["k_cache"].shape
    page = shape[1] * shape[2]
    backing = torch.empty(
        page * (shape[0] + 1) + 128, device=flaggems_vllm.device, dtype=torch.uint8
    )
    strides = (page, shape[2], 1)
    if invalid == "token_stride":
        strides = (page + 64, shape[2] + 1, 1)
    elif invalid == "page_stride":
        strides = (page - 2, shape[2], 1)
    else:
        strides = (page + 1, shape[2], 1)
    inputs["k_cache"] = backing.as_strided(shape, strides)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@pytest.mark.parametrize(
    "even,odd,cosine,sine,dimension,value",
    [
        (-0.5, 2.015625, 0.5443320274353027, 0.8388698697090149, 449, 0.6796875),
        (
            0.671875,
            -1.359375,
            0.9095131158828735,
            -0.4156752824783325,
            448,
            0.046142578125,
        ),
    ],
)
def test_mthreads_rope_fma_order(
    cache_format, even, odd, cosine, sine, dimension, value
):
    if flaggems_vllm.vendor_name != "mthreads":
        pytest.skip("MUSA writer's even-cos and odd-sin FMA order")
    inputs = _make_inputs(1, 1, 32, cache_format, scale_value=1.0)
    inputs["kv"].zero_()
    inputs["kv"][:, 448::2] = even
    inputs["kv"][:, 449::2] = odd
    inputs["cos_sin_cache"][:, :32].fill_(cosine)
    inputs["cos_sin_cache"][:, 32:].fill_(sine)
    expected = inputs["k_cache"].clone()
    TORCH_REF(**dict(inputs, k_cache=expected))
    if cache_format == "bf16":
        assert expected[0, 0, dimension].item() == value
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)
    _assert_bytes_equal(inputs["k_cache"], expected)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
@pytest.mark.parametrize("invalid", ["kv_dtype", "cache_width", "positions", "scale"])
def test_invalid_metadata_is_rejected(invalid):
    inputs = _make_inputs(4, 4, 32, "fp8")
    if invalid == "kv_dtype":
        inputs["kv"] = inputs["kv"].float()
    elif invalid == "cache_width":
        inputs["k_cache"] = inputs["k_cache"][:, :, :511]
    elif invalid == "positions":
        inputs["position_ids"] = inputs["position_ids"][:3]
    else:
        inputs["fp8_scale"] = None
    with pytest.raises(RuntimeError):
        flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
def test_reference_encoder_matches_cpu_at_fp8_midpoints():
    # CPU-native FP8 is independent of both software encoders. In particular,
    # 1.0625 / 0.125 must remain exactly 8.5 and round to the even lattice point.
    lattice = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = (lattice[:-1] + lattice[1:]) * 0.5
    below = torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf")))
    above = torch.nextafter(midpoints, torch.full_like(midpoints, float("inf")))
    positive = torch.cat((lattice, midpoints, below, above))
    values = torch.cat((positive, -positive))
    expected = values.to(torch.float8_e4m3fn).view(torch.uint8)
    actual = encode_e4m3(values.to(flaggems_vllm.device)).cpu()
    assert torch.equal(actual, expected)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
def test_thead_encoder_all_finite_fp16_values():
    if flaggems_vllm.vendor_name not in ("thead", "mthreads"):
        pytest.skip("Backend E4M3 conversion sweep")
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    values = values[torch.isfinite(values)].float()
    count = values.numel() // 32
    slots = torch.arange(count, device=flaggems_vllm.device)
    inputs = _make_inputs(count, count, 32, "fp8", scale_value=1.0, slot_mapping=slots)
    inputs["kv"].zero_()
    inputs["kv"][:, 448::2] = 1
    inputs["position_ids"] = slots
    table = torch.cat((values.reshape(count, 32), torch.zeros(count, 32)), dim=1)
    inputs["cos_sin_cache"] = table.to(flaggems_vllm.device)
    native_cache = inputs["k_cache"].clone()
    if VLLM_REF is not None:
        VLLM_REF(**dict(inputs, k_cache=native_cache))
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)
    if VLLM_REF is not None:
        _assert_bytes_equal(inputs["k_cache"], native_cache)
    actual = inputs["k_cache"].cpu().view(torch.uint8).reshape(-1, 512)
    expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(actual[:count, 448::2].reshape(-1), expected)


@pytest.mark.fused_deepseek_v4_kv_rope_insert
def test_thead_ue8m0_all_positive_finite_bf16_maxima():
    if flaggems_vllm.vendor_name not in ("thead", "mthreads"):
        pytest.skip("Backend integer UE8M0 exponent sweep")
    maxima = (
        torch.arange(0x7F80, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    )
    count = (maxima.numel() + 6) // 7
    padded = torch.zeros(count * 7, dtype=torch.bfloat16)
    padded[: maxima.numel()] = maxima
    slots = torch.arange(count, device=flaggems_vllm.device)
    inputs = _make_inputs(count, count, 32, "packed_v4", slot_mapping=slots)
    values = torch.zeros(count, 512, dtype=torch.bfloat16)
    values[:, :448] = (
        padded.reshape(count, 7, 1).expand(count, 7, 64).reshape(count, 448)
    )
    inputs["kv"] = values.to(flaggems_vllm.device)
    inputs["cos_sin_cache"][:, :32].fill_(1)
    inputs["cos_sin_cache"][:, 32:].zero_()
    native_cache = inputs["k_cache"].clone()
    if VLLM_REF is not None:
        VLLM_REF(**dict(inputs, k_cache=native_cache))
    flaggems_vllm.fused_deepseek_v4_kv_rope_insert(**inputs)
    if VLLM_REF is not None:
        _assert_bytes_equal(inputs["k_cache"], native_cache)
    pages = inputs["k_cache"].cpu().reshape(-1, 32 * 584)
    codes = pages[:, 32 * 576 :].reshape(-1, 8)[:count, :7].reshape(-1)
    exponent = torch.ceil(torch.log2(maxima.float().clamp_min(1e-4) / 448))
    assert torch.equal(codes[: maxima.numel()], (exponent + 127).to(torch.uint8))
    if flaggems_vllm.vendor_name == "mthreads":
        padded_exponent = torch.ceil(torch.log2(padded.float().clamp_min(1e-4) / 448))
        expected = encode_e4m3(padded.float() * torch.exp2(-padded_exponent))
        expected = expected.reshape(count, 7, 1).expand(count, 7, 64)
        payload = pages[:, : 32 * 576].reshape(-1, 576)[:count, :448]
        assert torch.equal(payload, expected.reshape(count, 448))
