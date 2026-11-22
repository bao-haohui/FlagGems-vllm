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

import os

import pytest
import torch

import flaggems_vllm
from baseline.fused_deepseek_v4_kv_rope_insert import (
    CACHE_FORMATS,
    OP_NAME,
    build_inputs,
    hygon_baseline,
    mthreads_baseline,
    native,
)

from . import base

VLLM_REF = native(flaggems_vllm.device)
BENCHMARK_FORMATS = CACHE_FORMATS
if flaggems_vllm.vendor_name == "mthreads":
    # The common benchmark uses torch.cuda event/graph names. Match the
    # repository's MUSA functional harness through the vendor adapter.
    import torchada  # noqa: F401

    VLLM_REF = mthreads_baseline
    BENCHMARK_FORMATS = ("packed_v4",)
elif flaggems_vllm.vendor_name == "hygon":
    VLLM_REF = (
        hygon_baseline if os.environ.get("KV_ROPE_HYGON_NATIVE_LIBRARY") else None
    )
    BENCHMARK_FORMATS = ("packed_v4",)

pytestmark = [
    pytest.mark.fused_deepseek_v4_kv_rope_insert,
    pytest.mark.skipif(VLLM_REF is None, reason="vLLM native op unavailable"),
]


class FusedDeepseekV4KVRopeInsertBenchmark(base.Benchmark):
    def __init__(self, cache_format):
        super().__init__(OP_NAME, VLLM_REF, [torch.bfloat16], is_inplace=True)
        self.cache_format = cache_format
        self.set_gems(flaggems_vllm.fused_deepseek_v4_kv_rope_insert)

    def get_input_iter(self, dtype):
        for num_tokens, num_tokens_insert, cache_block_size in self.shapes:
            binding = {
                "num_tokens": num_tokens,
                "num_tokens_insert": num_tokens_insert,
                "cache_block_size": cache_block_size,
                "cache_format": self.cache_format,
            }
            # Match PR #1083: zero cache and one masked slot when M > 1.
            args, _ = build_inputs(binding, dtype, self.device, cache_fill=0)
            yield args


@pytest.mark.parametrize("cache_format", BENCHMARK_FORMATS)
def test_fused_deepseek_v4_kv_rope_insert(cache_format):
    torch.manual_seed(0)
    bench = FusedDeepseekV4KVRopeInsertBenchmark(cache_format)
    label = f"vLLM native ({flaggems_vllm.vendor_name})"
    if flaggems_vllm.vendor_name == "mthreads":
        label = "official vllm-musa Triton (zero Q)"
    elif flaggems_vllm.vendor_name == "hygon":
        label = "official LightOp HIP (zero Q, safe positions)"
    print("baseline:", label, "cache_format:", cache_format)
    original_str = base.BenchmarkResult.__str__

    def native_label(result):
        return original_str(result).replace(
            "Torch Latency (ms)", f"{label} Latency (ms)"
        )

    base.BenchmarkResult.__str__ = native_label
    try:
        bench.run()
    finally:
        base.BenchmarkResult.__str__ = original_str
