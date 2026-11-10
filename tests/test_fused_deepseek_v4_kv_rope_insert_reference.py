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

"""CPU oracle checks; runnable with --noconftest without Triton or an accelerator."""

import math

import pytest
import torch

from baseline.fused_deepseek_v4_kv_rope_insert import (
    CACHE_FORMATS,
    build_inputs,
    encode_e4m3,
    torch_reference,
)

pytestmark = pytest.mark.fused_deepseek_v4_kv_rope_insert


def test_encoder_all_finite_bf16_values():
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = values.view(torch.bfloat16).float()
    values = values[torch.isfinite(values)]
    expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(encode_e4m3(values), expected)


def test_encoder_fp8_midpoints_and_neighbors():
    lattice = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    ties = (lattice[:-1] + lattice[1:]) * 0.5
    below = torch.nextafter(ties, torch.full_like(ties, -float("inf")))
    above = torch.nextafter(ties, torch.full_like(ties, float("inf")))
    positive = torch.cat((lattice, ties, below, above))
    values = torch.cat(
        (positive, -positive, torch.tensor([0.0, -0.0, float("inf"), -float("inf")]))
    )
    expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(encode_e4m3(values), expected)


def test_rope_pair_order_and_position_lookup():
    kv = torch.arange(1024).reshape(2, 512).to(torch.bfloat16)
    before = kv.clone()
    cache = torch.zeros(1, 2, 512, dtype=torch.bfloat16)
    table = torch.zeros(2, 64)
    table[0, :32] = 1
    table[1, 32:] = 1
    torch_reference(kv, cache, torch.tensor([1, 0]), torch.tensor([1, 0]), table, 2)
    expected = before[0].clone()
    expected[448::2] = -before[0, 449::2]
    expected[449::2] = before[0, 448::2]
    assert torch.equal(cache[0, 1], expected)
    assert torch.equal(cache[0, 0], before[1])
    assert torch.equal(kv, before)


def _native_fp8_bytes(values):
    return values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def _identity_expected(args):
    """Build expected physical bytes using scalar offsets and CPU-native casts."""
    kv, cache, slots, _, _, block_size, scale, mxfp8 = args
    raw = cache.view(torch.uint8)
    pages = raw.as_strided((cache.shape[0], raw.stride(0)), (raw.stride(0), 1))
    expected = pages.clone()
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        block, offset = divmod(slot, block_size)
        if cache.dtype == torch.bfloat16:
            start = offset * raw.stride(1)
            expected[block, start : start + 1024] = kv[row].view(torch.uint8)
        elif cache.dtype == torch.float8_e4m3fn:
            start = offset * raw.stride(1)
            expected[block, start : start + 512] = _native_fp8_bytes(
                kv[row].float() * (1.0 / scale)
            )
        else:
            group_size, quantized_dims = (32, 512) if mxfp8 else (64, 448)
            data_bytes, scale_bytes = (512, 16) if mxfp8 else (576, 8)
            codes = []
            for column in range(0, quantized_dims, group_size):
                group = kv[row, column : column + group_size].float()
                exponent = math.ceil(
                    math.log2(max(group.abs().max().item(), 1e-4) / 448)
                )
                quantized = _native_fp8_bytes(group * (2.0**-exponent))
                start = offset * data_bytes + column
                expected[block, start : start + group_size] = quantized
                codes.append(exponent + 127)
            if not mxfp8:
                start = offset * data_bytes + 448
                expected[block, start : start + 128] = kv[row, 448:].view(torch.uint8)
                codes.append(0)
            start = block_size * data_bytes + offset * scale_bytes
            expected[block, start : start + scale_bytes] = torch.tensor(
                codes, dtype=torch.uint8
            )
    return pages, expected


@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@pytest.mark.parametrize("padded_pages", [False, True])
def test_physical_layout_and_untouched_bytes(cache_format, padded_pages):
    binding = dict(
        num_tokens=5, num_tokens_insert=3, cache_block_size=4, cache_format=cache_format
    )
    args, _ = build_inputs(
        binding,
        torch.bfloat16,
        "cpu",
        padded_pages=padded_pages,
        slot_mapping=torch.tensor([7, -9, 0]),
        scale_value=0.7,
    )
    args[0].copy_(torch.linspace(-64, 64, 5 * 512).reshape(5, 512))
    args[4][:, :32] = 1
    args[4][:, 32:] = 0
    before = args[0].clone()
    pages, expected = _identity_expected(args)
    assert torch_reference(*args) is None
    assert torch.equal(pages, expected)
    assert torch.equal(args[0], before)


@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
@pytest.mark.parametrize("empty", [False, True])
def test_empty_and_all_negative_do_not_touch_cache(cache_format, empty):
    count = 0 if empty else 3
    args, _ = build_inputs(
        dict(
            num_tokens=count,
            num_tokens_insert=count,
            cache_block_size=4,
            cache_format=cache_format,
        ),
        torch.bfloat16,
        "cpu",
        slot_mapping=torch.full((count,), -7, dtype=torch.int64),
    )
    args[4].resize_(0, 64)
    before = args[1].view(torch.uint8).clone()
    assert torch_reference(*args) is None
    assert torch.equal(args[1].view(torch.uint8), before)
