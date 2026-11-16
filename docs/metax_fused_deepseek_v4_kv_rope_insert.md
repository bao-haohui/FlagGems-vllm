# MetaX DeepSeek V4 KV RoPE 插入实现

## 当前状态

2026-11-16，分支 `codex/metax-fused-deepseek-v4-kv-rope-insert`，从现有
`b232a69fc6c0b4ff3d4f7151c0d8cb171ec79ef7` 派生，保留已有 NVIDIA/Ascend 实现。
MetaX 实现、后端导出、tuning 配置、功能测试和原生 benchmark 接入已完成。
MC550 真实编译与设备功能 108/108、CPU 参考 24/24、三方对拍 34/34 通过；
11月15日的性能结果保留为历史记录；正式性能口径已按参考 PR 调整为仓库
benchmark 的 Kernel / CUDA Graph 双模式和逐用例加速比算术平均。
模型端到端及其他 MetaX 设备尚未验证。

实测生产代码为 `11dd7b7`，kernel 文件 SHA256：
`7edec4bc18dc598115bd60eb4f0465c936a1f8bfad4982a4811fd4b59a8591a2`。

参考 [PR #1083](https://github.com/flagos-ai/FlagGems-vllm/pull/1083) 的融合数据流
和 [PR #849](https://github.com/flagos-ai/FlagGems-vllm/pull/849) 的厂商后端覆盖方式。
遵循 `AGENTS.md` 和 `workflow.md`；本 checkout 没有协议引用的
`optimization.md` / `deep_opt.md`，以下记录路径、候选配置和验收口径。

## 接口与数值规则

接口和 vLLM v0.32.0 的 `fused_deepseek_v4_kv_rope_insert` 一致：

```python
flaggems_vllm.fused_deepseek_v4_kv_rope_insert(
    kv, k_cache, slot_mapping, position_ids, cos_sin_cache,
    cache_block_size, fp8_scale=None, kv_mxfp8=False,
)
```

`kv` 为只读连续 BF16 `[N,512]`，仅处理 INT64 `slot_mapping[M]` 指定的前 M 行，
M≤N。INT64 `position_ids[N]` 与 FP32 `cos_sin_cache[P,64]` 同设备，均连续。
位置表排列为 32 个 cos 后接 32 个 sin；只旋转最后 64 维，相邻两个元素为一对。
输入已经准备好 KV，本算子不做归一化、不处理 Q、不计算 attention。
缓存原地更新，返回 `None`，不分配中间 Tensor。

| 缓存路径 | 逻辑形状/类型 | 舍入与物理布局 |
|---|---|---|
| BF16 | `[B,S,512]` BF16 | FP32 RoPE 后转 BF16，保留真实 token/page stride |
| per-tensor FP8 | `[B,S,512]` E4M3FN | FP32 RoPE 直接乘标量倒数、饱和到 ±448、RNE；无中间 BF16 舍入 |
| V4 packed | `[B,S,584]` uint8 | 每 token 448B NoPE FP8 + 128B RoPE BF16；7 个 group-64 UE8M0 scale + 1 个零字节 |
| MXFP8 | `[B,S,528]` uint8 | RoPE 先舍入 BF16；全 512 维 group-32 FP8，16 个 UE8M0 scale |

Packed 页内先存整页所有 payload，再存整页所有 scale，不能按每行 584B/528B 的
普通 AoS 记录理解。V4 的 scale 起点为 `page + S*576`，MXFP8 为 `page + S*512`。
不同页可以存在 padding；完整 backing 中未写 token、页间 padding 和前后缀均应不变。

分组 scale 为 `2**ceil(log2(max(amax,1e-4)/448))`，存储指数加 127 的 UE8M0。
保留 E4M3FN 的 RNE、饱和、负零及两种 FP8 路径不同的 BF16 舍入位置。
RoPE 两个分量都显式融合 cos 乘积：

```text
even_out = fma(even, cos, -round_fp32(odd*sin))
odd_out  = fma(odd,  cos,  round_fp32(even*sin))
```

8 个能区分融合顺序的定向输入在原生 BF16、V4 packed 和 per-tensor FP8 上确认了
偶数分量的融合顺序；奇数分量另有独立反例。MXFP8 与 V4 使用相同的原生旋转代码。
MetaX 奇数分量与 NV 初版融合 `even*sin` 的顺序不同。反例 `even=-0.5, odd=2.015625,
cos=0.5443320274353027, sin=0.8388698697090149` 在两种顺序下分别舍入为
BF16 的 `0.67578125` 与 `0.6796875`。MetaX 独立 Torch reference 使用
`_rope_fma_on_cos=True`；其余平台保持原有默认值。
量化路径另外显式保留缩放前的符号位，保证精确负零与舍入到零的负数均存为负 FP8 零。
当前 FlagTree/MetaX 工具链的向量乘法、浮点取负可能将零规范成正零；已观察到
16 元素乘法探针保留负零，而 512 元素探针丢失符号。RoPE 的正弦乘积使用带乘积
符号零的 FMA 表达等价乘法，取负使用 IEEE 符号位 XOR；避免依赖易变的隐式融合。

有效 slot 必须唯一且在缓存范围内，有效 position 必须在表内。负 slot 不读取对应
KV、position 和位置表，也不写缓存；相应 position 可无效，全负 slot 时位置表可为空。
M=0 直接返回。
不读取 M 之后的 DP padding 行。输入与缓存应互不别名；重复有效 slot 不保证确定性。

本阶段面向模型的有限输入、有限 RoPE 结果和正有限 FP8 scale；非有限中间结果、
训练/autograd、非法有效索引不在当前验收范围。Host 不复制 GPU 索引到 CPU 做值检查。

## Host 路由与 kernel

Host 先复用通用输入元数据检查，再执行 MetaX 布局检查，最后由 `CACHE_KIND`
生成四种编译分支。BF16/FP8 支持无重叠的 token/page stride；packed 要求页内连续，
V4 另外要求页起点 BF16 对齐。不支持的布局明确抛出 `NotImplementedError`。

所有非空路径均为一次 Triton launch：

1. 一个 program 处理 `BLOCK_TOKENS` 行，先读取 slot 并生成 live mask。
2. 对 position、KV、cos/sin 的读取应用 live mask；尾块和负 slot 均不访问无效地址。
3. 在寄存器中计算 RoPE，根据格式执行量化和分页写入。
4. 页偏移使用 INT64，维度索引保持 INT32。

相较 NV 初版的一个 program/token，候选多 token tile 用于降低大 M 的 CTA 调度成本。
MetaX 的线程宽度不同，warp 数通过设备测试选择，不直接认定 NV 参数最优。
分组量化通过 FP32 指数位计算向上取整的二次幂 scale，并直接构造倒数，省去
`log2/exp2`。由于分组输入均来自 BF16，归约后的所有有限 amax 对应范围可穷举核查。
这属于数值等价及性能候选改写，设备收益须实测。

生产路径只使用 Torch dtype/shape/stride/device/storage_offset 元数据和设备上下文；
数值计算及类型转换全部由 Triton 完成，无 Torch compute fallback，无外部库生产路径。
公共导出由现有后端替换机制选择 `._metax.ops.`，其他平台实现不受该覆盖影响。

## Autotune

配置名为 `fused_deepseek_v4_kv_rope_insert`，仅新增到 MetaX 的 `tune_configs.yaml`。
`@libentry()` + `@libtuner()` 使用如下候选：

| BLOCK_TOKENS | num_warps | num_stages |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 1 | 1 |
| 4 | 1 | 1 |
| 4 | 2 | 1 |
| 8 | 4 | 1 |
| 16 | 4 | 1 |

FlagTree AABS 还会按实际 M 调整配置；本轮小 M 实测最优配置额外出现
`BLOCK_TOKENS=1/2, num_warps=4`。6 个基础配置和这 2 个实测派生配置均单独核查：
8 配置 × 四格式的随机/尾块 32 项通过，8 配置 × 四格式 × 四相位的零值 128 项通过。
配置记录应同时保存 YAML 候选和运行时实际选择，不能只从 YAML 推断发射参数。

Key 为实际插入行数 M、缓存格式、页大小、page stride、token stride，另外遵循
libtuner 的 dtype key。候选使用相同数学路径，通过 CUDA Graph 计时选择。
首次编译和调优必须放在正式性能计时/图捕获之前；有效 slot 唯一且重复写入幂等，
调优不需要清零或恢复缓存。此后相同 key 复用缓存配置。
NV 仍保留原有固定配置及其 no-autotune 豁免，本改动不改变 NV 调度。

## 原生基线与运行方式

对比基线为 MetaX 官方仓库 `MetaX-MACA/vLLM-metax` 固定提交
`03950d148de3da946b8706ddee1aaee5de2e94a9` 的独立同名入口，计算文件
`csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` 未修改。
不构造 Q、不执行额外 Q/KV 归一化。官方 TU 与头文件可以保持原样，使用私有
Torch namespace 单独注册，避免覆盖厂商安装库。

测试/benchmark 支持显式指定该私有原生库，例如：

```bash
export DNN_VENDOR=metax
export KV_ROPE_NATIVE_LIBRARY=/path/to/official_native_baseline.so
export KV_ROPE_NATIVE_NAMESPACE=my_native_namespace
export TRITON_PRINT_AUTOTUNING=1
python -m pytest -q tests/test_fused_deepseek_v4_kv_rope_insert.py --quick
python -m pytest -q tests/test_fused_deepseek_v4_kv_rope_insert.py
for mode in kernel cudagraph; do
  python -m pytest -q -s benchmark/test_fused_deepseek_v4_kv_rope_insert.py \
    --level comprehensive --mode "$mode" --warmup 1000 --iter 100 \
    --parallel 0 --record json --output "/path/to/private/benchmark-$mode.json"
done
```

显式选择的库或 namespace 不可用时直接报错；不会静默改用 Torch 计时。
benchmark 和 PR1083 一样在无原生入口时跳过，性能路径只接受原生基线。
Torch reference 仍可用于精度核查。基线/库注册仅存在于 baseline、tests、benchmark，
生产实现不会导入这些模块。

正式比较要求两者使用同一设备、Torch、MACA 和计时模式。已建立的旧开发环境
Torch2.8/MACA3.7 与原生基线 Torch2.10/MACA3.9 不能直接比较；本轮以独立 venv
在原生 Torch2.10/MACA3.9 容器内安装指定 FlagTree，保留两套旧环境。
实际统一栈：`Torch 2.10.0+metax3.9.0.8`、MACA3.9、
`FlagTree 0.7.0.post2+metax3.6`（导入的 Triton 模块3.6.0）、FlagGems5.4.0。
厂商原生库 SHA256 始终为
`95fb06a4eb94be7db4cb1821d095c191e3e924697a319468c06c66046768fe71`。

原生 kernel 在检查负 slot 前读取位置表，因此“跳过行 position 无效/表为空”的扩展
安全测试使用独立 reference，或将原生安全输入适配成本计入完整接口计时。
核心性能集合所有 position 合法，直接调用原生入口。

## 验收与当前记录

性能 active set 固定为 `benchmark/core_shapes.yaml` 中该算子的 10 组 shape × 四格式，
S=32、M=1/2/4/8/64/100/1024/2048/8192/16384，1024 行用例有 N=1027 的 DP padding。
采用原生延迟/Triton 延迟，单位 ms；对每格式10个逐用例比值和总体40个比值
分别取算术平均，同时报告最小值和未达 0.9 的 shape。
不以总体平均掩盖单项回归。模型端到端性能不等于算子 CUDA Graph 延迟。

### 对齐参考 PR 的正式测量口径（2026-11-16）

PR1083 固定版本 `9a7624f35f1dfde3c1891b660ccd32443c174b5f`，PR849 固定版本
`e1d87fcdd7d5725af325e7034ebd3f9e9c3f644f`。代码沿用 Apache 头、Black/isort/flake8、
后端 override 导出和 `base.Benchmark` 子类；MetaX 的 tile/autotune 与数值兼容修正
服务于本设备和原生数值契约，保留已验证实现。未把另一个算子的固定发射参数照搬过来。

- 输入与 PR1083 保持同一分布/布局：BF16 KV，零初始化缓存，乱序唯一 slot；
  M>1 时最后一个 slot 为 -1，M=1 时有效。位置为 0..N-1，S=32，FP8 scale=0.5。
  不再为性能测量强制全部 slot 有效。保留 uint8 分配后 view，以避开厂商 FP8 cast 依赖。
- 每个格式测试开始前固定 `torch.manual_seed(0)`，两种模式使用相同构造流程。
  功能测试继续默认使用 0xA5 哨兵缓存，完整 backing 的越界/漏写检查保留。
- 正式计时全部由仓库 pytest benchmark 完成，`--level comprehensive`，
  该算子 comprehensive 和 core 均为10组 shape；`--parallel 0` 串行。
- Kernel 使用 `triton.testing.do_bench`，warmup=1000ms、rep=100ms、median；
  CUDA Graph 使用 `do_bench_cudagraph`，rep=100ms、median，warmup 由该 helper 管理。
  Graph 分支不读取 CLI 的 `--warmup`，不能将其描述成 Graph 额外预热1000ms。
- 同一输入先由原生计时，再由 Triton 计时，沿用仓库顺序；输入、JIT、调优、库加载
  不计入正式延迟。两者共用相同 Torch/MACA 栈，未修改全局 benchmark 计时实现。
- 独立脚本只做40组完整缓存字节对拍、KV只读核查和 JSON 后处理，不执行正式计时。
  后处理必须验证两份 JSON 各有4格式/40条成功记录，使用 `mean(native_ms/triton_ms)`，
  不使用几何平均，也不使用 `mean(native_ms)/mean(triton_ms)`。

### 正式重测结果（2026-11-16）

benchmark 代码为 `a78e59d`，生产 kernel 与原生库哈希均保持不变。
本地及服务器 CPU 同组36/36通过，新增benchmark输入40/40完整缓存字节和KV只读对拍通过；
Kernel / CUDA Graph 各4个pytest格式测试通过、各40条成功记录，执行exit0。
两模式共80条性能记录，全部高于0.9；没有触发进一步性能优化。模型端到端未运行。
新调优结果额外选择 `BLOCK_TOKENS=1, num_warps=2, num_stages=1`；在11月15日
已验证的8配置之外，该新配置另做四格式随机/尾块4项和四相位零值16项，全部通过。

| 格式 | Kernel 算术平均 | Kernel 最小 | CUDA Graph 算术平均 | CUDA Graph 最小 |
|---|---:|---:|---:|---:|
| BF16 | 1.493× | 1.333× | 1.484× | 1.185× |
| per-tensor FP8 | 1.780× | 1.365× | 2.203× | 1.283× |
| V4 packed | 2.173× | 2.000× | 2.864× | 2.245× |
| MXFP8 | 2.095× | 1.900× | 2.720× | 1.982× |
| 总体 | 1.885× | 1.333× | 2.318× | 1.185× |

下表沿用相同40组输入，展示两种计时模式的逐用例延迟和加速比。延迟单位 μs；
完整精度保存在仓库benchmark输出的两份原始JSON，表中显示值仅作小数舍入。

| 格式 | N / M | Kernel 原生 | Kernel Triton | Kernel 加速比 | Graph 原生 | Graph Triton | Graph 加速比 |
|---|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1 / 1 | 9.216 | 6.912 | 1.333× | 3.004 | 1.780 | 1.688× |
| BF16 | 2 / 2 | 9.216 | 6.912 | 1.333× | 3.027 | 1.990 | 1.521× |
| BF16 | 4 / 4 | 9.472 | 6.912 | 1.370× | 3.135 | 2.025 | 1.548× |
| BF16 | 8 / 8 | 9.728 | 6.912 | 1.407× | 3.411 | 2.048 | 1.665× |
| BF16 | 64 / 64 | 10.496 | 7.168 | 1.464× | 3.506 | 2.247 | 1.560× |
| BF16 | 100 / 100 | 10.752 | 7.168 | 1.500× | 3.505 | 2.126 | 1.649× |
| BF16 | 1027 / 1024 | 18.432 | 9.216 | 2.000× | 4.920 | 3.432 | 1.434× |
| BF16 | 2048 / 2048 | 19.968 | 11.776 | 1.696× | 6.466 | 4.711 | 1.373× |
| BF16 | 8192 / 8192 | 32.000 | 21.504 | 1.488× | 20.387 | 16.730 | 1.219× |
| BF16 | 16384 / 16384 | 45.824 | 34.304 | 1.336× | 35.205 | 29.717 | 1.185× |
| per-tensor FP8 | 1 / 1 | 13.824 | 6.912 | 2.000× | 6.965 | 1.990 | 3.499× |
| per-tensor FP8 | 2 / 2 | 14.080 | 7.680 | 1.833× | 6.945 | 2.462 | 2.821× |
| per-tensor FP8 | 4 / 4 | 13.824 | 7.680 | 1.800× | 7.029 | 2.867 | 2.452× |
| per-tensor FP8 | 8 / 8 | 14.336 | 7.680 | 1.867× | 7.230 | 2.845 | 2.542× |
| per-tensor FP8 | 64 / 64 | 14.336 | 7.936 | 1.806× | 7.304 | 2.976 | 2.454× |
| per-tensor FP8 | 100 / 100 | 14.592 | 7.936 | 1.839× | 7.342 | 2.996 | 2.450× |
| per-tensor FP8 | 1027 / 1024 | 22.272 | 11.008 | 2.023× | 9.682 | 5.733 | 1.689× |
| per-tensor FP8 | 2048 / 2048 | 25.856 | 14.592 | 1.772× | 12.571 | 8.473 | 1.484× |
| per-tensor FP8 | 8192 / 8192 | 47.872 | 32.000 | 1.496× | 35.652 | 26.330 | 1.354× |
| per-tensor FP8 | 16384 / 16384 | 73.728 | 54.016 | 1.365× | 61.810 | 48.190 | 1.283× |
| V4 packed | 1 / 1 | 17.152 | 7.680 | 2.233× | 10.211 | 2.764 | 3.694× |
| V4 packed | 2 / 2 | 17.408 | 8.448 | 2.061× | 10.229 | 3.321 | 3.080× |
| V4 packed | 4 / 4 | 17.408 | 8.448 | 2.061× | 10.223 | 3.332 | 3.068× |
| V4 packed | 8 / 8 | 17.408 | 8.448 | 2.061× | 10.343 | 3.380 | 3.061× |
| V4 packed | 64 / 64 | 17.408 | 8.704 | 2.000× | 10.422 | 3.454 | 3.017× |
| V4 packed | 100 / 100 | 17.664 | 8.704 | 2.029× | 10.469 | 3.524 | 2.970× |
| V4 packed | 1027 / 1024 | 28.928 | 11.520 | 2.511× | 15.937 | 5.865 | 2.717× |
| V4 packed | 2048 / 2048 | 35.072 | 14.848 | 2.362× | 21.946 | 8.671 | 2.531× |
| V4 packed | 8192 / 8192 | 78.848 | 35.584 | 2.216× | 66.265 | 29.510 | 2.245× |
| V4 packed | 16384 / 16384 | 135.936 | 61.952 | 2.194× | 123.050 | 54.598 | 2.254× |
| MXFP8 | 1 / 1 | 16.896 | 7.424 | 2.276× | 9.834 | 2.498 | 3.937× |
| MXFP8 | 2 / 2 | 17.152 | 8.192 | 2.094× | 9.844 | 3.160 | 3.115× |
| MXFP8 | 4 / 4 | 17.152 | 8.448 | 2.030× | 9.856 | 3.374 | 2.921× |
| MXFP8 | 8 / 8 | 17.152 | 8.448 | 2.030× | 9.971 | 3.418 | 2.917× |
| MXFP8 | 64 / 64 | 17.152 | 8.448 | 2.030× | 10.067 | 3.456 | 2.913× |
| MXFP8 | 100 / 100 | 17.408 | 8.704 | 2.000× | 10.095 | 3.565 | 2.832× |
| MXFP8 | 1027 / 1024 | 28.160 | 11.776 | 2.391× | 15.257 | 6.469 | 2.358× |
| MXFP8 | 2048 / 2048 | 33.792 | 15.360 | 2.200× | 20.975 | 9.362 | 2.240× |
| MXFP8 | 8192 / 8192 | 76.032 | 38.144 | 1.993× | 63.131 | 31.758 | 1.988× |
| MXFP8 | 16384 / 16384 | 131.840 | 69.376 | 1.900× | 117.507 | 59.300 | 1.982× |

### 11月15日的功能和优化历史

功能测试补充负 slot 配无效 position/空表、全部 FP8 midpoint 的 BF16 邻值、UE8M0
边界、负零、饱和、非零 cache storage offset、非法 packed 布局，以及 per-tensor FP8
不允许中间 BF16 舍入的定向样例。现有测试继续覆盖 DP padding、页大小32/256、
大 M、乱序 slot、页间 padding、token stride、KV 不变和返回 None。

| 检查 | 本轮结果 |
|---|---|
| 独立 CPU reference、显式原生入口选择与两种 FMA 定向例 | 24 项通过 |
| scale 指数改写的 CPU 数学核查 | 32,640 种非负有限 BF16 位模式全部一致 |
| Python 语法及本次9个文件的全部 pre-commit hooks | 通过 |
| Python wheel 构建 | 通过；不是设备 kernel 编译 |
| MC550 四路径编译/精度 | 108/108 通过 |
| 原生/Triton/独立 CPU 三方对拍 | 34/34 通过，含完整 backing、页 padding、KV 不变 |
| 基础与实测 AABS 配置随机尾块核查 | 32/32 通过 |
| 配置级旋转零值回归 | 4 格式 × 4 相位 × 8 配置，128/128 通过 |
| 设备 profiler | BF16 2048行，5次原生、5次SUT各捕获5个目标kernel，一次API一次launch |
| 同栈原生性能与 best config | 40/40 达标，详见下表 |
| 仓库 benchmark 入口 smoke | 4个格式测试、40条core记录通过；原生标签正确 |
| 模型端到端 | 未运行 |

本地 CPU 为 macOS ARM，CPU reference 测试和 wheel 构建不能替代 MetaX JIT 验证。

| Trial | 假设/改变 | 验证 | 性能/决定 |
|---|---|---|---|
| 初版 | NV融合数据流、多token tile、指数位scale | 初轮54/59通过，发现FMA和零符号差异 | 精度未过，不计时 |
| 精度收敛 | 两cos FMA、乘积符号零、位运算取负和量化前符号保留 | 108项功能、34组三方、32配置随机、128配置零值全部通过 | keep |
| 配置选择 | 六组基础候选及AABS实测派生配置 | 所有40条性能输入先与原生字节比较 | 全部达标，keep |

以下性能表为11月15日的历史口径，不能与上面的正式口径混合汇总。
当时使用独立脚本 `triton.testing.do_bench_cudagraph`，每轮30ms、三轮取中位数，
交替测量顺序，slot 全部有效，缓存以0xA5初始化。
原生和 SUT 置于同一进程、同一 Torch/MACA 栈；JIT、autotune、输入构造和库加载在计时外。
下面的几何平均对每格式10个M等权，最小值单独报告；全部 shape 的原生/SUT 延迟和
实际 best config 已保存。本表中的大M延迟为 M=N=16384，单位 μs。

| 格式 | 几何平均加速比 | 最小加速比 | 大M原生 μs | 大M Triton μs |
|---|---|---|---|---|
| BF16 | 1.410× | 1.169× | 35.154 | 30.066 |
| per-tensor FP8 | 2.053× | 1.283× | 61.925 | 48.257 |
| V4 packed | 2.766× | 2.244× | 123.039 | 54.343 |
| MXFP8 | 2.586× | 1.994× | 117.958 | 59.170 |

固定单token/一warp发射对照中，per-tensor FP8 在 M=16384 为54.715μs，
当前选择为48.257μs；其余格式的配置收益按具体shape判断，完整结果保留原始采样。
没有 shape 低于0.9，未触发 deep optimization；后续应以模型实际shape分布和端到端结果
决定进一步优化目标。这里不将 CUDA Graph 算子加速比推算为模型加速比。

若关键 shape 加速比低于 0.9，先核对原生入口、调度、autotune 和计时方式，
再结合 profiler 或编译产物、launch 数、配置交叉点证据记录进一步优化；不得跳过精度检查。
