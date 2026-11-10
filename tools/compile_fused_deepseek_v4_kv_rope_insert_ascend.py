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

"""Compile every Ascend cache branch without executing the target Triton kernel.

Requires a configured Linux CANN/Triton Ascend environment and an NPU for
runtime discovery and input allocation. This is not a correctness/perf test.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("bf16", "fp8", "packed_v4", "mxfp8"),
        default=("bf16", "fp8", "packed_v4", "mxfp8"),
    )
    parser.add_argument("--block-sizes", type=int, nargs="+", default=(32, 64))
    parser.add_argument("--tokens", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(s <= 0 for s in args.block_sizes) or (
        args.tokens is not None and any(n <= 0 for n in args.tokens)
    ):
        parser.error("block sizes and token counts must be positive")
    if os.getenv("TRITON_INTERPRET", "0") not in ("0", ""):
        parser.error("unset TRITON_INTERPRET: interpreter execution is not compilation")

    # Running a script sets sys.path[0] to tools/, not the checkout root.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.environ.setdefault("DNN_VENDOR", "ascend")
    import torch
    import torch_npu

    torch_npu.npu.set_device(args.device)
    import flaggems_vllm
    from baseline.fused_deepseek_v4_kv_rope_insert import build_inputs
    from flaggems_vllm.ops.fused_deepseek_v4_kv_rope_insert import _validate_inputs
    from flaggems_vllm.runtime.backend._ascend.ops.fused_deepseek_v4_kv_rope_insert import (
        _cache_views,
        _fused_deepseek_v4_kv_rope_insert_kernel,
        _get_vector_core_count,
    )

    if "._ascend.ops." not in (
        flaggems_vllm.fused_deepseek_v4_kv_rope_insert.__module__
    ):
        raise RuntimeError("public API did not resolve to the Ascend override")
    core_count = _get_vector_core_count(args.device)
    counts = args.tokens or (1, core_count * 2 + 3)
    versions = {}
    for name in ("torch", "torch-npu", "triton", "triton-ascend", "flagtree"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {
        "validation": "Ascend kernel compilation only; no target kernel execution",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "vector_cores": core_count,
        "device": torch_npu.npu.get_device_name(args.device),
        "cases": [],
    }
    for cache_format in args.formats:
        for block_size in args.block_sizes:
            for count in counts:
                case = dict(
                    cache_format=cache_format, block_size=block_size, tokens=count
                )
                try:
                    inputs, _ = build_inputs(
                        dict(
                            num_tokens=count + 3,
                            num_tokens_insert=count,
                            cache_block_size=block_size,
                            cache_format=cache_format,
                        ),
                        torch.bfloat16,
                        f"npu:{args.device}",
                        padded_pages=True,
                    )
                    kind = _validate_inputs(*inputs)
                    kv, cache, slots, positions, table, _, scale, _ = inputs
                    byte_cache, bf16_cache = _cache_views(kv, cache, kind)
                    programs = min(count, core_count)
                    compiled = _fused_deepseek_v4_kv_rope_insert_kernel.warmup(
                        kv,
                        byte_cache,
                        bf16_cache,
                        slots,
                        positions,
                        table,
                        scale,
                        count,
                        byte_cache.stride(0),
                        byte_cache.stride(1),
                        CACHE_BLOCK_SIZE=block_size,
                        CACHE_KIND=kind,
                        NUM_PROGRAMS=programs,
                        GROUP_SIZE=32 if kind == 3 else 64,
                        grid=(programs,),
                    )
                    if compiled is None or not hasattr(compiled, "asm"):
                        raise RuntimeError("compiler did not return a compiled kernel")
                    case.update(
                        status="passed",
                        hash=str(getattr(compiled, "hash", "")),
                        artifacts=sorted(compiled.asm),
                    )
                except Exception:
                    case.update(status="failed", error=traceback.format_exc())
                report["cases"].append(case)
    report["passed"] = all(c["status"] == "passed" for c in report["cases"])
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
