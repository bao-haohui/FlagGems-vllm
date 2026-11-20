# MThreads DeepSeek V4 KV RoPE cache insert

## 范围与接口

对应vLLM0.32的`fused_deepseek_v4_kv_rope_insert`，用于DeepSeek V4的SWA KV缓存写入。
本实现位于`runtime/backend/_mthreads/ops/fused_deepseek_v4_kv_rope_insert.py`，
由平台导出覆盖公共`flaggems_vllm.fused_deepseek_v4_kv_rope_insert`。

KV输入为只读、连续BF16 `[N,512]`；只处理前M行，其中M为slot个数、M<=N。
int64 slot/position与FP32 `[P,64]` cos/sin必须在同一MUSA设备。
cache原地更新，返回None；没有Q、RMSNorm或新建输出KV。
负slot在读取position和cos/sin前跳过；M=0直接返回。
活跃slot需唯一且在缓存范围内，对应position有效。

| cache | RoPE及量化 | 物理布局 |
|---|---|---|
| BF16 / 512元素 | 最后64维RoPE后舍入BF16 | 实参page/token stride |
| per-tensor E4M3 / 512元素 | FP32 RoPE直接乘逆scale再转换，避免BF16中间舍入 | 实参page/token stride |
| V4 uint8 / 584B | NoPE448按64分组FP8；RoPE64存BF16 | 页内S×576B数据，其后S×8B scale |
| MXFP8 uint8 / 528B | RoPE先BF16舍入，全512维按32分组FP8 | 页内S×512B数据，其后S×16B scale |

支持cache页padding和非零storage offset；普通格式支持不重叠token stride。
packed要求页内记录紧密，V4 page stride/offset需要BF16对齐。
不满足布局约束时显式报错，无Torch计算或外部库回退。
本轮验收覆盖有限KV/cos-sin和常规模型范围的正有限per-tensor scale；
未验收NaN/Inf量化、scale倒数溢出或RoPE/中间计算溢出。

## 实测环境

| 项目 | 固定环境 |
|---|---|
| 设备 / 编译目标 | MTT S5000 80GiB，`musa arch=31, warp_size=32` |
| 驱动 / MUSA SDK | 3.3.5-server / 5.2.0 |
| Torch / torch_musa | 2.11.0.post2+musa5.2.0 |
| vLLM / vllm-musa | 0.32.0 / 0.1.32 |
| 私有FlagTree / 实际Triton模块 | 0.7.0+mthreads3.6 / 3.6.0 |
| FlagGems | `a3d9230612dfcff4d4bd68fe22d5e00e84a08371` |

一个个人容器暴露所有设备，默认`MUSA_VISIBLE_DEVICES=-1`；执行前在宿主选择空闲卡，
以每进程环境变量限定一张，再通过Runtime API核对物理PCIe地址与进程内`musa:0`。
所有设备阶段串行，不为每张卡创建容器，也不改共享驱动、基础包或他人任务。

## Triton数据流与数值顺序

一个program处理B个token，构造`[B,512]` tile，grid为`ceil(M/B)`。
通过constexpr CACHE_KIND选择存储路径：读取slot并建立live mask，读取有效position/cos/sin，
执行相邻维GPT-J RoPE，量化，计算分页地址并masked store。
V4数据区与scale区独立计算地址，7个有效UE8M0 scale之后的padding字节明确写零。
输入KV不写回，跳过的slot、page padding和offset外哨兵保持原值。
当B=1时，slot使用标量读取，负slot在向量工作之前直接退出。
B>1时按行mask，保留混合有效/无效token及末块的正确语义。

RoPE按官方MUSA writer的显式顺序：

```text
even_out = fma(even, cos, -(odd * sin))
odd_out  = fma(even, sin, odd * cos)
```

乘积先FP32舍入，再进入指定FMA；关闭隐式融合，避免编译器改变舍入顺序。
当前编译器将浮点一元取负降成`+0-x`，会把部分数学负零变正零。
SUT通过FP32位模式异或`0x80000000`取负，保留符号，独立CPU参考保留严格字节标准。
这是针对已观察编译行为的正确性处理，不作为额外性能收益。

## 优化与适用范围

| 实现点 | 具体方法 | 类型与对比边界 |
|---|---|---|
| 融合 | RoPE、scale求解、FP8转换和分页写入在同一kernel完成，无中间tensor/workspace | 通用融合；官方V4基线也已融合，不能将单kernel本身算成相对它新增的收益 |
| 多token分块 | B改变每program工作量和grid数，减少大M下的program调度；小M保留B=1候选 | 通用调度设计，候选由S5000实测选择 |
| warp联合调优 | B与W一起测量，兼顾并行度、寄存器和归约开销 | S5000平台配置；W不要求等于B，实际lane分配由Triton布局决定 |
| UE8M0算术 | 从amax的FP32指数及尾数阈值0.75构造幂指数，位构造倒数scale，替代div/log2/exp2 | 通用精确幂变换；官方MUSA writer已有同类优化，不单独归因加速 |
| 原生E4M3转换 | 使用当前MUSA后端支持的`tl.float8e4nv`，位视图写cache | 已验证工具链能力，未使用PPU的软件编码器或假定其他芯片可用 |
| 消除冗余FP8符号修补 | 已验原生转换保留正负零，删除额外的逐元素符号提取与恢复 | 依据当前MUSA工具链实测；RoPE取负的位翻转仍保留 |
| 负slot提前退出 | B=1先做uniform标量slot判断，直接结束跳过的program | 通用无效工作消除，保持空cos/sin表和非法position跳过语义 |
| 分组量化限幅消除 | 精确向上取整幂scale保证有限BF16组内幅值不超过448，去掉该路径min/max | 通用数学优化；外部per-tensor scale没有同样保证，继续限幅 |
| 编译期格式分流 | CACHE_KIND constexpr消除其他格式分支；V4独立NoPE与BF16 RoPE写入 | 通用专门化，不引入运行时格式切换kernel |

本实现没有手工编写MUSA汇编、改变共享驱动或依赖未测的芯片特性。
S5000的32-lane warp、FP8转换支持及实测最优配置是平台依据；算法本身主要来自通用Triton优化。

调优名`fused_deepseek_v4_kv_rope_insert`位于`_mthreads/tune_configs.yaml`，
通过`runtime.get_tuned_config`、`libtuner`和`libentry`接入。
候选(B,W)为(1,1)、(1,2)、(1,4)、(2,1)、(4,1)、(4,2)、(8,4)、(16,4)，num_stages=1。
key包含M、CACHE_KIND、page size及page/token stride，首次调用调优，后续复用缓存。
稳定调用发射一个kernel；首次调优的多次试运行需提前完成。

FlagTree3.6默认开启AABS：当B大于M时，可将B缩小为不小于M的2的幂，保留原W。
因此M=1时也可能测到由(B=8,W=4)派生的(B=1,W=4)。YAML是初始候选集，
报告中的命中配置以运行时实际结果为准。lane如何承担B行数据由编译布局决定。
8个候选并不表示B=W：B描述数据tile和grid，W描述每program的线程资源。
归约、布局转换、内存合并与资源占用共同决定时延，靠实测选择，不能只按token数指定warp数。

该算子提供专用MUSA autotune计时回调，直接使用`torch.musa.MUSAGraph/Event`，
不依赖pytest/torchada对CUDA名字的兼容重定向。回调测量已有kernel，不实现目标数值计算。
独立Python、全新调优缓存下的四格式冷启动是单独验收项。
仓内benchmark继续使用原Benchmark类；仅在MThreads benchmark进程导入torchada，
让通用计时器的CUDA接口名称落到MUSA实现，生产算子没有全局CUDA补丁。

## 基线与性能方法

V4使用官方`MooreThreads/vllm-musa` Triton writer，零Q `[M,0,512]`，
`q_head_padded=0`、`apply_q_norm=False`，直接调用并丢弃空Q返回值。
官方grid `(M,q_blocks+1)`在零Q时仍保留KV program，没有额外真实Q计算。
该writer文件SHA256锁定为
`3fcb87272adac3975d3c5dad292a3e815dd72ab1345d125e0533e79dd3d19f6c`，
与[官方固定提交](https://github.com/MooreThreads/vllm-musa/blob/45341a3a3499e51a39dcd47da3d54adb2b43f958/vllm_musa/kernels/deepseek_v4_rope_kv_insert.py)一致。
这是厂商Triton基线，不是MUSA C++或Torch计算基线。
BF16/per-tensor FP8/MXFP8尚缺匹配官方基线，只做正确性验证，不报告其官方加速比。

沿用PR #1083/#849的仓内Benchmark类与pytest入口，seed0、零cache、M>1最后slot=-1，
十个标准shape、BF16输入和S32；1024插入档N=1027，其余N=M。
分别使用`kernel`和`cudagraph`模式（MUSA运行栈），warmup1000ms、rep100ms、parallel0。
加速比=官方writer延迟/SUT延迟；主统计为各点等权算术平均，几何平均作为补充，逐点列出回归。
CPU参考、编译、输入生成及首次autotune不作为稳定算子延迟。

官方基线已有RoPE BF16负零变正零的已知差异，原始基线严格检查为22/26。
SUT不复制该行为。额外三方与benchmark输入检查对SUT始终要求完整字节一致；
官方差异若出现，逐项核实仅为已知RoPE零符号，保留差异计数。

## 验证状态

2026-11-20，第一版功能验证通过，性能状态为 **limited / needs_deep_opt**。
最终保留trial6；它包含原生FP8转换、B/W联合调优、负slot提前退出和分组量化去冗余限幅。
S5000完整功能110通过、8项其他平台专用检查跳过；CPU参考39通过。
独立新进程、无CUDA兼容补丁、冷调优缓存下四格式smoke4/4；
38项布局/边界、8候选×4格式的32项随机检查及128项候选负零检查通过，严格比对完整storage。
实际benchmark工厂40组输入均与独立CPU参考一致，V4同时对比固定官方writer。

数值覆盖包含63,488个有限FP16输入值的E4M3转换；32,640个非负有限BF16组最大值的
UE8M0 scale和V4 payload检查。CPU另外穷举相同最大值，验证位域指数和倒数精确、
最大归一化幅值为448。测试参考仍显式限幅，不依赖被测kernel的去限幅实现。

设备profiler实际采集到每次SUT与官方调用各一个kernel，另有24条显式候选计时。
该短采样和候选计时是分析证据，主要性能结果以仓内完整benchmark为准。
四项静态CI、Black/isort/flake8和无Torch计算回退审计纳入最终交付检查。
模型端到端、其他MThreads型号和NaN/Inf量化未在本轮验收。

匹配MUSA/FlagTree环境激活并在Python启动前选好空闲卡后，沿用仓内入口：

```bash
python -B -m pytest -q tests/test_fused_deepseek_v4_kv_rope_insert.py
python -B -m pytest --noconftest -q tests/test_fused_deepseek_v4_kv_rope_insert_reference.py
python -B -m pytest -q -s benchmark/test_fused_deepseek_v4_kv_rope_insert.py \
  --level comprehensive --mode kernel --warmup 1000 --iter 100 --parallel 0 \
  --record json --output benchmark-kernel.json
python -B -m pytest -q -s benchmark/test_fused_deepseek_v4_kv_rope_insert.py \
  --level comprehensive --mode cudagraph --warmup 1000 --iter 100 --parallel 0 \
  --record json --output benchmark-cudagraph.json
```

## 优化试验与交付边界

trial0为起点；trial1删除已证明多余的FP8符号修补，SUT Kernel逐点延迟降低约4%—10%。
trial2/3的小M独立kernel和该路径隐式FP融合均通过精度，但主要性能回归，已整体回退。
trial4补B1/W2、B1/W4，trial5加负slot标量提前退出，trial6仅移除UE8M0组内冗余限幅。
候选、计时协议、源码与逐轮keep/revert均记录，未通过改基线或放松字节精度获得收益。

trial7将V4 B1表达为1D数据、2D归约；完整功能、候选、输入及短profile通过。
共享设备多次全卡繁忙，未执行该版完整active-set benchmark，因此将该实验留档并恢复已测trial6。
这不表示已证明扁平布局更慢。它没有进入本次生产实现。

最终代码没有Torch计算回退。`view(torch.uint8)`是无拷贝类型视图，shape/stride/numel/
storage_offset是元数据；MUSA Graph/Event/synchronize只用于首次autotune测量。
所有RoPE、归约、量化及写cache均在Triton中完成。未支持的布局报错，不转Torch计算。
无新增workspace、autograd承诺或其他芯片型号的性能结论。

最终性能仍有Kernel关键点未达0.9，且存在基线异常点，**不能宣称整体性能验收完成**。
仓库引用的`optimization.md`、`deep_opt.md`缺失，已按workflow的深入优化步骤保留
dispatch/计时/候选审计、device profile、数值证明和逐轮试验结论。

## 最终保留版本的性能记录

仅V4 packed，baseline/SUT，延迟μs；最终源码SHA256为
`b73a14492615555be77dc1c671c9968dfacbdadc27636ccde0a4d4ed671dd314`。

| 模式 | 算术平均 | 几何平均 | 最小 | 最大 | <0.9点数 |
|---|---:|---:|---:|---:|---:|
| kernel | 1.700×* | 1.320× | 0.858× | 4.855× | 3 |
| cudagraph | 1.056× | 1.048× | 0.917× | 1.331× | 0 |

*Kernel前两项官方基线21.44/22.72μs异常偏高，抬高了1.700×均值；该均值不能作为稳定总体加速结论。
Graph单轮十点均>=0.9；Kernel的M8/M64/M8192仍<0.9。本轮状态为limited / needs_deep_opt。
共享设备未锁频或独占预约，数值正确性与性能验收分别判断。

| N / M / S | Kernel 官方 | Kernel SUT | 加速比 | Graph 官方 | Graph SUT | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 1 / 32 | 21.440 | 4.920 | 4.358× | 5.606 | 5.555 | 1.009× |
| 2 / 2 / 32 | 22.720 | 4.680 | 4.855× | 5.526 | 5.644 | 0.979× |
| 4 / 4 / 32 | 6.240 | 4.840 | 1.289× | 7.110 | 5.610 | 1.267× |
| 8 / 8 / 32 | 4.360 | 5.080 | 0.858× | 5.453 | 5.598 | 0.974× |
| 64 / 64 / 32 | 4.640 | 5.160 | 0.899× | 5.321 | 4.817 | 1.104× |
| 100 / 100 / 32 | 4.600 | 5.000 | 0.920× | 5.269 | 4.889 | 1.078× |
| 1027 / 1024 / 32 | 6.040 | 6.200 | 0.974× | 6.450 | 6.989 | 0.923× |
| 2048 / 2048 / 32 | 9.440 | 9.760 | 0.967× | 8.653 | 9.434 | 0.917× |
| 8192 / 8192 / 32 | 22.800 | 25.360 | 0.899× | 23.938 | 24.583 | 0.974× |
| 16384 / 16384 / 32 | 41.800 | 42.780 | 0.977× | 38.257 | 28.751 | 1.331× |

这两个模式的原始JSON、完整逐点比值和输入/源码哈希均保留；未删除异常点或只报告加速点。

V4实际命中配置（本轮缓存结果，后续设备状态变化可能重新选择）：

| M | B | W |
|---:|---:|---:|
| 1 | 1 | 4 |
| 2 | 1 | 1 |
| 4 | 1 | 1 |
| 8 | 1 | 1 |
| 64 | 1 | 1 |
| 100 | 1 | 1 |
| 1024 | 1 | 1 |
| 2048 | 8 | 4 |
| 8192 | 4 | 2 |
| 16384 | 4 | 2 |
