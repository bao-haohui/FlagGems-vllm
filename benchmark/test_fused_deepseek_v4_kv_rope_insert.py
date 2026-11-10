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
    OP_NAME,
    build_inputs,
    native,
    torch_reference,
)

from . import base

VLLM_REF = native(flaggems_vllm.device)

pytestmark = [
    pytest.mark.fused_deepseek_v4_kv_rope_insert,
]


class FusedDeepseekV4KVRopeInsertBenchmark(base.Benchmark):
    def __init__(self, cache_format):
        reference = VLLM_REF or functools.partial(
            torch_reference, _assume_all_valid=True
        )
        super().__init__(OP_NAME, reference, [torch.bfloat16], is_inplace=True)
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
            # The Torch graph baseline uses fixed-shape indexing. General
            # functional tests separately cover negative slots and padding.
            slots = torch.randperm(num_tokens_insert, device=self.device)
            args, _ = build_inputs(binding, dtype, self.device, slot_mapping=slots)
            yield args


@pytest.mark.parametrize("cache_format", CACHE_FORMATS)
def test_fused_deepseek_v4_kv_rope_insert(cache_format):
    bench = FusedDeepseekV4KVRopeInsertBenchmark(cache_format)
    print("baseline:", "vLLM native CUDA" if VLLM_REF else "Torch unfused reference")
    if VLLM_REF is None:
        bench.run()
    else:
        original_str = base.BenchmarkResult.__str__

        def native_label(result):
            return original_str(result).replace(
                "Torch Latency (ms)", "vLLM CUDA Latency (ms)"
            )

        base.BenchmarkResult.__str__ = native_label
        try:
            bench.run()
        finally:
            base.BenchmarkResult.__str__ = original_str
