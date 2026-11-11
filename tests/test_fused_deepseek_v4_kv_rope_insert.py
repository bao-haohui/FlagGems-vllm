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

import pytest
import torch

import flaggems_vllm
from baseline.fused_deepseek_v4_kv_rope_insert import (
    CACHE_FORMATS,
    build_inputs,
    encode_e4m3,
    native,
    torch_reference,
)
from flaggems_vllm import testing as utils

from . import conftest as cfg

VLLM_REF = native(flaggems_vllm.device)
REFERENCE = VLLM_REF or torch_reference

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
def test_public_api_selects_ascend_override():
    if flaggems_vllm.vendor_name != "ascend":
        pytest.skip("Ascend backend dispatch check")
    assert "._ascend.ops." in (
        flaggems_vllm.fused_deepseek_v4_kv_rope_insert.__module__
    )


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
