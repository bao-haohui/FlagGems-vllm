# 平头哥 DeepSeek V4 KV RoPE 插入实现

2026-11-19，分支 `codex/thead-fused-deepseek-v4-kv-rope-insert`，基于679d18ad。
ZW810E四格式实现、后端导出、autotune、功能验证及原生性能对比已完成。
标准Kernel/CUDA Graph各40组全部超过0.9，逐用例加速比算术平均分别2.476×/2.638×。
模型端到端及其他PPU型号尚未验证。

最终实测kernel SHA256：
`6af42bc0e2bf654c9830337deffbf6815f392626c1a75ee6b6d604d95cf6ee46`。正式数据来自最终格式整理后的源码；
初轮结果单独留档，不混入本文统计。

## 实现与集成

沿用 [PR1083](https://github.com/flagos-ai/FlagGems-vllm/pull/1083) 固定版本
`9a7624f35f1dfde3c1891b660ccd32443c174b5f` 的接口、缓存协议和性能输入；
参考 [PR849](https://github.com/flagos-ai/FlagGems-vllm/pull/849) 固定版本
`e1d87fcdd7d5725af325e7034ebd3f9e9c3f644f` 的厂商override、公共导出和标准Benchmark接入方式。
遵循AGENTS.md/workflow.md；当前checkout没有其引用的optimization.md/deep_opt.md，
路径与性能active set在实现前单独冻结，首版已达到性能门槛，未触发后续deep opt。

- 生产文件：[_thead/ops/fused_deepseek_v4_kv_rope_insert.py](../src/flaggems_vllm/runtime/backend/_thead/ops/fused_deepseek_v4_kv_rope_insert.py)。
- 后端导出：[_thead/ops/__init__.py](../src/flaggems_vllm/runtime/backend/_thead/ops/__init__.py)。
- 调优配置：[_thead/tune_configs.yaml](../src/flaggems_vllm/runtime/backend/_thead/tune_configs.yaml)。
- 复用已有公共API、共享参数校验、算子元数据、[功能测试](../tests/test_fused_deepseek_v4_kv_rope_insert.py)
  和[标准benchmark](../benchmark/test_fused_deepseek_v4_kv_rope_insert.py)，未改其他平台kernel。
- 当前厂商Triton3.5缺少TLE；保留已验证的BMM可选导入保护，缺少TLE时BMM显式unsupported。
  该导入兼容修正不改BMM的3个kernel，也不参与本算子计算。

接口为 `flaggems_vllm.fused_deepseek_v4_kv_rope_insert(...) -> None`。
输入连续BF16 KV `[N,512]`，前M行由INT64 slot `[M]` 指定写入槽位，M≤N；
INT64 position `[N]` 索引FP32 `[P,64]` 的cos/sin表。只旋转末64维GPT-J相邻维；
输入KV只读、缓存原地更新，不计算Q、KVNorm或attention。

| 格式 | 缓存类型/宽度 | 数值和布局 |
|---|---|---|
| BF16 | BF16 / 512 | FP32旋转后BF16存储，保留真实token/page stride |
| per-tensor FP8 | E4M3FN / 512 | FP32旋转直接乘标量倒数后饱和编码，无中间BF16舍入 |
| V4 packed | uint8 / 584 | NoPE448按64分组FP8、RoPE64为BF16；7个UE8M0及1个零padding |
| MXFP8 | uint8 / 528 | 旋转后先舍入BF16，全512维按32分组FP8、16个UE8M0 |

packed每页先放所有token的payload，再放所有scale，不能按584/528字节AoS行处理。
支持页间padding、非零storage offset；plain格式允许不重叠token stride。
packed页内必须连续，V4页/offset须BF16对齐；不支持的布局抛NotImplementedError。
有效slot须唯一且在范围内、有效position须合法；负slot在访问position/table前屏蔽，
全负slot允许空表。M=0校验参数后返回，无kernel。

生产路径只用Tensor元信息及FP8缓存的无拷贝uint8 view，不调用Torch计算或数值转换，
不分配中间Tensor、不调用官方库作fallback。面向有限模型输入、有限RoPE结果及正有限scale；
非有限中间结果、非法有效索引、重复slot、重叠别名和autograd不在当前验收范围。

## Triton数据流与优化

每个program处理 `[B,512]` 的token tile，grid为ceil(M/B)。先加载slot形成live mask，
再读取有效position、位置表和KV。将64维RoPE拆成32对，计算后按cache kind编译期分流：
直接BF16存储、标量FP8编码、NoPE group64量化或全维group32量化，并直接写最终分页缓存。

1. **融合数据流（通用）**：RoPE、量化、scale及分页地址计算在同一kernel内完成。
   官方基线本来也是一个融合kernel，所以不能将本次加速单独归因于减少其kernel数量。
2. **B/W联合调优（通用方法、平台实测配置）**：使用libentry/libtuner和后端YAML，
   搜索 `(B,W)=(1,1),(2,1),(4,1),(4,2),(8,4),(16,4)`，stage固定1。
   key包含M、格式、页大小和两个stride；libtuner还加入Tensor dtype。
   参数由实测选择，Triton编译器据此安排线程布局，没有固定每个warp必须对应一个token。
3. **UE8M0整数指数构造（通用数学改写）**：设absmax的FP32指数域为E、尾数域为m，
   `ceil(log2(absmax/448))+127 = E-8+(m>0x600000)`。
   absmax先clamp到1e-4；从code构造 `2**(-exponent)`，省去div/log2/ceil/exp2。
   前提是本路径的有限BF16来源及正常FP32 absmax，已覆盖全部非负有限BF16最大值。
4. **E4M3字节编码（当前PPU编译器适配）**：复用现有PPU量化算子的FP32位模式RNE encoder，
   普通数通过指数重偏置和mantissa ties-to-even舍入，次正规数通过FP32 magic-add舍入。
   饱和到±448，用uint8写入；不生成float8e4nv指针或FP8 cast，也不经过FP16。
   这不表示PPU硬件没有FP8能力；官方SDK的原生转换在本机能够运行。

实际40组中W全部选1，B多数选1。B2命中：BF16的M2048/8192/16384、MXFP8的M16384；
其余场景使用B1。
其余候选仍经过独立验证，不能把“支持多token调优”写成“所有场景采用多token”。
候选编译元信息可出现shared memory，例如B16/W4为2048B，不能把无中间全局Tensor
等同于没有编译器使用的shared memory。未做单项消融，不分摊各优化对整体加速的贡献。

## PPU原生数值对齐

当前固定PPU原生构建在两个分量都收缩cos乘积：

```text
even_out = fma(even, cos, -round_fp32(odd*sin))
odd_out  = fma(odd,  cos,  round_fp32(even*sin))
```

初始NV表达在8192行BF16上与原生差11个字节；两组定向输入也区分了BF16、
packed RoPE及per-tensor FP8舍入。两cos FMA的独立CPU参考与原生完全一致。
测试使用 `_rope_fma_on_cos=True`，不据此声称与NV所有临界值逐位相同。

另以显式带符号零FMA实现正弦乘法、IEEE符号位翻转实现取负，并在FP8编码时
保留缩放前符号位，消除工具链对零表达式的折叠差异。它们是数值兼容所需，
没有把初次失败结果计为通过，也没有放宽完整缓存按字节比较的标准。

## 基线与运行环境

设备：ZW810E、97920MiB；本轮逻辑cuda0对应物理PPU0。
初检PPU1有作业，未干预，改用空闲PPU0；双方在同一卡、同一软件栈串行执行。
Ubuntu24.04.2 / Python3.12.3 / Torch2.9.0 / 厂商Triton3.5.0+ppu2.0.0.oe，
驱动1.3.2-d7f5a2，SDK `/usr/local/PPU_SDK`。兼容target为cuda/arch80/warp32，实际为PPU。

官方来源：[t-head/vllm-sail](https://github.com/t-head/vllm-sail/tree/2610096f4393ffd8c7cb45ee3defb9c85d1e6c74)，
固定2610096，device代码源自vLLM0.31。共享vLLM0.13没有相应入口；原官方计算片段
逐字节保留，使用个人SDK/Torch2.9 ABI桥接注册到 `_bao_ppu_baseline`。
它不是Torch参考，也不表示完整新版插件或vLLM0.32整模已安装。

| 原生构件 | SHA256 |
|---|---|
| 官方计算私有库 | `5f9da16a2a4313ec1390579aed4fbbc4e977af5b20f2cb06134d9896fd90594c` |
| 私有Torch入口 | `83ace69c0fbafbee521920c9e760c1c875f8e8ba7bff55443372d90586e29b3a` |

原生库本轮未改。Kernel模式包含各自host调用/调度路径影响；CUDA Graph减少这部分影响。
私有host桥接与完整厂商绑定存在接口耗时边界，结果应解释为当前同栈固定基线的比较。

## 功能与性能结果

| 验证 | 结果 |
|---|---|
| 服务器完整设备pytest | 110/110 |
| CPU参考测试 | 本地及服务器同组36/36，不叠加计数 |
| Triton、官方计算、CPU参考三方对拍 | 34/34，含stride、padding、负slot、DP padding和空输入 |
| 所有候选随机及尾块 | 24/24 |
| 所有候选正负零 | 96/96 |
| 实际标准benchmark输入 | 40/40，完整缓存字节相同、KV只读、None返回 |
| 编码/scale专项 | 63488个有限FP16值编码、32640个非负有限BF16最大值；属于pytest专项覆盖 |

上述集合有重叠，不累加成一个不同输入总数。三方对拍中8项原生端含安全position适配，
4项为空输入；Triton直接在kernel内屏蔽无效position。性能输入的position全部有效，
原生直接入口不加这层适配。

标准active set：每格式M1/2/4/8/64/100/1024/2048/8192/16384，S32，
M1024对应N1027，其余N=M。每格式seed0、cache清零、M>1末slot=-1、scale0.5。
未改benchmark的get_latency，使用Kernel/CUDA Graph两模式，warmup1000/iter100（ms参数），
parallel0、median。实际次数由所用厂商Triton helper估算；其源码快照已保存。
加速比为逐case `native_ms/triton_ms` 的等权算术平均，不是平均延迟之比或几何平均。

| 格式 | Kernel均值 | Kernel最小 | CUDA Graph均值 | CUDA Graph最小 |
|---|---:|---:|---:|---:|
| BF16 | 1.458× | 1.141× | 1.589× | 1.392× |
| per-tensor FP8 | 2.899× | 2.038× | 3.092× | 2.527× |
| V4 packed | 2.885× | 1.954× | 3.133× | 2.338× |
| MXFP8 | 2.660× | 1.774× | 2.740× | 2.118× |
| 总体40组 | 2.476× | 1.141× | 2.638× | 1.392× |

两模式各4个pytest测试、40条成功记录，无低于0.9项。首版已达标，不继续无依据扩大搜索。
Torch profiler另外采集了5次原生和5次Triton调用，trace中各5个计算kernel；
BF16 M2048样例GPU均值约3.448us/2.312us，仅作为融合/调度证据，不混入正式统计。
独立nsys/ncu等工具未找到；已有实际device trace、候选元信息及命中配置。

## 复现与审计

```bash
ppu-smi -i 0
source /root/baohaohui/ppu-kv-rope/implementation/scripts/activate.sh
cd /root/baohaohui/FlagGems-vllm
bash "$KV_ROPE_PPU_IMPL/scripts/run-validation.sh"
bash "$KV_ROPE_PPU_IMPL/scripts/run-benchmark.sh"
```

执行前确认所选卡空闲。验证runner的设备阶段串行，记录源码、配置、原生库及日志。
标准pytest性能命令为：

```bash
python -B -m pytest -q -s benchmark/test_fused_deepseek_v4_kv_rope_insert.py \
  --level comprehensive --mode kernel --warmup 1000 --iter 100 \
  --parallel 0 --record json --output /your/output/benchmark-kernel.json
# 第二次将mode改为cudagraph，并使用独立输出文件。
```

服务器记录位于 `/root/baohaohui/ppu-kv-rope/implementation`；工作区本地资料位于
`fused_deepseek_v4_kv_rope_insert/ppu_implementation`，包括设计、原始JSON、逐例表、
profiler trace、编译元信息、失败轮次和环境审计。未升级共享Torch/Triton/SDK/驱动，
所有源码、缓存和产物均在本人目录；仅使用一条SSH，没有重置设备或停止他人进程。
