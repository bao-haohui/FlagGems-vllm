# Ascend: fused_deepseek_v4_kv_rope_insert

Initial implementation for Ascend 910B4/A2; 910C/A3 needs separate validation.
The vLLM v0.32.0 contract and the NVIDIA implementation in PR #1083 are the
semantic reference. PR #849 supplies the core-bound, strided-task launch pattern.

Reference heads: #1083 `9a7624f35f1dfde3c1891b660ccd32443c174b5f`;
#849 `e1d87fcdd7d5725af325e7034ebd3f9e9c3f644f`. The fork base is
`025f3181e19283885513255d143ca2ad55e8d897`. Only #1083's seven operator
files were imported; unrelated upstream changes were excluded.

## Design and scope

- Public signature and metadata validation are shared with PR #1083. Empty
  inserts return before launching. Only the first M rows are processed, and
  negative slots skip all table reads and writes. Live slots must be unique,
  in range, and have valid positions. Input KV is never modified.
- Dispatch on cache kind before compiling. BF16 and per-tensor FP8 use the
  actual block/token strides. Packed V4/MXFP8 use actual block strides and
  separate page data/scale regions, with exactly the NVIDIA byte layout.
- One program iterates over tokens spaced by the launch grid, bounded by the
  runtime vector-core count, cached per device. Dimension offsets stay INT32;
  token/page bases stay INT64. The cache allocation must have fewer than 2^31
  slots. Packed V4 requires a BF16-compatible, aligned no-copy cache view.
- RoPE and group maxima use FP32. Packed paths round RoPE to BF16 before
  quantizing; the per-tensor FP8 path retains the FP32 RoPE result.
- E4M3 bytes use software round-to-nearest-even conversion with finite
  saturation. UE8M0 uses FP32 exponent/mantissa bits to round the scale up
  to a power of two; normalization multiplies by its inverse.
- All production computation stays inside Triton. Torch calls in host dispatch
  are metadata access and no-copy dtype views. No Torch/Ascend C fallback.
- This first Ascend version fixes semantic group sizes (64/32) and a core-bound
  launch; there is no autotuner or claimed optimal tiling. The inherited NVIDIA
  mapping keeps PR #1083's documented no-autotune exemption.
- Finite model activations are the primary packed-format workload. Non-finite
  grouped quantization and cross-device FMA behavior need device investigation;
  no stronger bitwise equivalence claim is made without native comparisons.

## Validation boundaries

Initial development used macOS ARM without CANN/Triton Ascend or an NPU.
Python compilation, CPU reference checks, packaging, and linting are local
checks; subsequent Linux/NPU validation is reported separately below.

An independent Torch reference belongs only to `baseline/`; production code
never imports it. Device tests run against it when the CUDA native op is
unavailable. FP8 conversion in that reference uses a software byte encoder so
an NPU FP8 cast kernel is not required. Report a Torch baseline as Torch,
not as a fused Ascend C baseline.

The BF16 Ascend C combination (partial rotary mul + scatter) remains a separate
candidate baseline; it is not the production implementation and is not assumed
available in every server image.

## Local checks (2026-11-11)

Development used an isolated Python 3.12.3 / CPU Torch 2.15.0 environment,
with Black 24.8.0, isort 5.12.0 and flake8 7.1.0 matching repository hooks.
This CPU environment is not a recommendation for the NPU server stack.

- 19 CPU oracle tests passed. These cover all 65,280 finite BF16 values against
  CPU-native E4M3 conversion, FP8 ties and adjacent FP32 values, position/pair
  order, all four cache layouts, unchanged bytes, page padding and empty inserts.
- The changed Python files compile to Python code objects and parse with the
  Python 3.8 grammar. This is not Triton JIT/MLIR/device compilation.
- Black/isort/flake8, YAML parsing, exports and whitespace checks passed.
- A Python wheel builds and includes both the common implementation and the
  Ascend override. Building this pure-Python package does not compile kernels.
- Host production audit: device-property/context calls and no-copy dtype
  views only; there is no Torch numeric computation or reference import.
- At the end of this local pass, NPU compilation and device checks were still
  pending. macOS ARM had no compatible `triton-ascend` distribution in the
  tested package index, CANN toolchain, or configured Linux runtime.

CPU oracle checks can be reproduced without importing the accelerator package:

```sh
python -m pytest --noconftest -q \
  tests/test_fused_deepseek_v4_kv_rope_insert_reference.py
```

## Ascend 910B4 server checks (2026-11-11)

Validated with Ascend 910B4-1 (40 vector cores), driver 25.2.0, CANN 9.0.0,
Python 3.11.15, Torch 2.9.0+cpu, torch_npu 2.9.0.post2, FlagTree
0.7.0+ascend3.5 (Triton module 3.5.1), FlagGems 5.0.2, and cann-shmem 1.6.0.
Torch/NPU and CANN came from the existing container image; project dependencies
were installed in a separate venv. Only one physical NPU was selected.

- Public package dispatch selected the Ascend override, and a minimal Torch NPU
  tensor operation passed.
- Real Triton compilation passed 16/16 cases: four cache formats, S=32/64,
  and M=1/83. M=83 exercises the core-bound token loop with 40 vector cores.
  Compiled artifacts include `npubin`, `ttir`, `ttadapter`, `bcmlir`, and
  `mlirbc`; these are device compiler results, not Python bytecode checks.
- All 40 operator `--quick` cases passed (80.58 s). They retain byte-exact
  cache comparison, KV immutability, padded pages, negative slots, empty
  inserts, noncontiguous token strides, API selection, and invalid metadata.
- All 19 CPU oracle checks also passed on the server (9.16 s).

Device validation exposed two issues in the independent Torch reference.
NPU `exp2(-3)` returned `0.1249999925494194`, which changes exact E4M3 tie
rounding; integral powers are now constructed exactly and normalization uses
the reciprocal lattice spacing. NPU `signbit(-0.0)` also lost the sign, so the
reference reads the IEEE sign bit directly. A device regression checks all
finite positive FP8 lattice points, midpoints and adjacent FP32 values, plus
their negative counterparts, against CPU-native E4M3 conversion. The production
Triton kernels and byte comparison requirements were unchanged.

These checks establish compilation and the tested functional cases on this
910B4 stack. Full non-quick shapes, native CUDA cross-checks, 910C, non-finite
grouped quantization, microbenchmarks and model end-to-end performance remain
unverified. No speedup or performance acceptance result is claimed.

## Server validation commands

Use an existing compatible CANN / torch_npu / Triton Ascend or FlagTree
environment. Preserve its dependency versions; the editable install below
does not replace them. Run from this repository's root:

```sh
export DNN_VENDOR=ascend
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install --no-deps --no-build-isolation -e .

# 1. Compile all four kernel specializations, without executing this kernel.
python tools/compile_fused_deepseek_v4_kv_rope_insert_ascend.py \
  --output /tmp/kv_rope_ascend_compile.json

# 2. Check public backend selection, correctness, cache gaps and input immutability.
python -m pytest -q tests/test_fused_deepseek_v4_kv_rope_insert.py --quick
python -m pytest -q tests/test_fused_deepseek_v4_kv_rope_insert.py

# 3. Operator microbenchmarks; Ascend prints "Torch unfused reference" baseline.
python -m pytest -q -s benchmark/test_fused_deepseek_v4_kv_rope_insert.py \
  --level core --mode kernel
python -m pytest -q -s benchmark/test_fused_deepseek_v4_kv_rope_insert.py \
  --level core --mode operator
```

The compilation tool rejects `TRITON_INTERPRET` and calls the real Triton
`kernel.warmup`. Default cases compile S=32/64 and one-token/core-bound grids;
`--formats`, `--block-sizes`, and `--tokens` can narrow failures. A JSON success
means the server compiler returned compiled kernels, not that output or latency
was checked. Record actual CANN/driver versions alongside the report.

Correctness tests use native vLLM on CUDA and the independent Torch reference
on Ascend. They compare cache bytes, including untouched regions. Investigate
any RoPE/FMA or quantization mismatch before accepting performance results.
Benchmark inputs contain only valid unique slots so the unfused Torch path
has fixed shapes; correctness tests separately retain invalid/omitted rows.
Decode and prefill timing here measures this operator, not full model throughput.

The initial active set is the ten `(N,M,S)` shapes in
`baseline/fused_deepseek_v4_kv_rope_insert.py`, across all four cache formats;
the core shape list is registered in `benchmark/core_shapes.yaml`. Report
per-case milliseconds (lower is better) and baseline/implementation speedup
(higher is better). No device performance acceptance or profiling has happened.
Keep compile/correctness failures separate from benchmark results. The checkout
contains `workflow.md`; its referenced `optimization.md`/`deep_opt.md` are absent.
