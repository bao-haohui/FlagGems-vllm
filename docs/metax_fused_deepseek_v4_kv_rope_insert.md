# MetaX DeepSeek V4 KV RoPE 插入实现

## 当前状态

2026-11-15，分支 `codex/metax-fused-deepseek-v4-kv-rope-insert`，从现有
`b232a69fc6c0b4ff3d4f7151c0d8cb171ec79ef7` 派生，保留已有 NVIDIA/Ascend 实现。
MetaX 初版、后端导出、tuning 配置、功能测试和原生 benchmark 接入已完成。
设备编译、精度和性能结果待本轮 MC550 验证后补充，当前不宣称性能达标。

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
RoPE 偶数分量使用两个乘法后相减，奇数分量显式 `tl.fma`；launch 禁用隐式浮点融合，
避免编译器任意改变已约定的舍入点。最终必须以同栈原生逐字节对拍确认。

有效 slot 必须唯一且在缓存范围内，有效 position 必须在表内。负 slot 跳过全部读取和
写入；相应 position 可无效，全负 slot 时位置表可为空。M=0 直接返回。
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
python -m pytest -q benchmark/test_fused_deepseek_v4_kv_rope_insert.py \
  --level core --mode cudagraph --warmup 10 --iter 30 --record json \
  --output /path/to/private/results.json
```

显式选择的库或 namespace 不可用时直接报错；不会静默改用 Torch 计时。
MetaX benchmark 在无原生入口时跳过，避免把 Torch reference 的延迟误称为原生基线。
Torch reference 仍可用于精度核查。基线/库注册仅存在于 baseline、tests、benchmark，
生产实现不会导入这些模块。

正式比较要求两者使用同一设备、Torch、MACA 和计时模式。已建立的旧开发环境
Torch2.8/MACA3.7 与原生基线 Torch2.10/MACA3.9 不能直接比较；本轮以独立 venv
在原生 Torch2.10/MACA3.9 容器内安装指定 FlagTree，保留两套旧环境。

原生 kernel 在检查负 slot 前读取位置表，因此“跳过行 position 无效/表为空”的扩展
安全测试使用独立 reference，或将原生安全输入适配成本计入完整接口计时。
核心性能集合所有 position 合法，直接调用原生入口。

## 验收与当前记录

性能 active set 固定为 `benchmark/core_shapes.yaml` 中该算子的 10 组 shape × 四格式，
S=32、M=1/2/4/8/64/100/1024/2048/8192/16384，1024 行用例有 N=1027 的 DP padding。
采用原生延迟/Triton 延迟，单位 ms；汇总最小值、各格式几何均值和未达 0.9 的 shape。
不以总体平均掩盖单项回归。模型端到端性能不等于算子 CUDA Graph 延迟。

功能测试补充负 slot 配无效 position/空表、全部 FP8 midpoint 的 BF16 邻值、UE8M0
边界、负零、饱和、非零 cache storage offset、非法 packed 布局，以及 per-tensor FP8
不允许中间 BF16 舍入的定向样例。现有测试继续覆盖 DP padding、页大小32/256、
大 M、乱序 slot、页间 padding、token stride、KV 不变和返回 None。

| 检查 | 本轮结果 |
|---|---|
| 独立 CPU reference 与显式原生入口选择 | 22 项通过 |
| scale 指数改写的 CPU 数学核查 | 32,640 种非负有限 BF16 位模式全部一致 |
| Python 语法、Black/isort/flake8 | 通过；最终提交前复核 |
| Python wheel 构建 | 通过；不是设备 kernel 编译 |
| MC550 四路径编译/精度 | 待运行 |
| 同栈原生性能与 best config | 待运行 |
| 模型端到端 | 未运行 |

本地 CPU 为 macOS ARM，CPU reference 测试和 wheel 构建不能替代 MetaX JIT 验证。

| Trial | 假设/改变 | 验证 | 性能/决定 |
|---|---|---|---|
| 0 | NV 固定单 token 作为同栈起点 | 待 MC550 运行 | 未计时 |
| 1 | MetaX 多 token tile 与指数位 scale | CPU 数学核查通过；设备待验证 | 待原生对拍及计时 |

若关键 shape 加速比低于 0.9，先核对原生入口、调度、autotune 和计时方式，
再结合 profiler 或编译产物、launch 数、配置交叉点证据记录进一步优化；不得跳过精度检查。
