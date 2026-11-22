# 海光 fused_deepseek_v4_kv_rope_insert

目标设备BW1000/gfx936，DTK26.04、Torch2.9、FlagTree0.7.0+hcu3.6。
接口沿用vLLM0.32 / PR1083；工程组织、libentry/libtuner、YAML配置和标准pytest benchmark
沿用PR849及既有MetaX/PPU/MThreads适配。

## 实现与优化

生产计算全部使用Triton。一次kernel完成slot筛选、尾64维GPT-J RoPE、按格式量化和分页写入；
KV保持只读，缓存原地更新，返回None。没有Torch计算或外部计算库fallback。

四种格式分别为BF16、per-tensor OCP E4M3、V4 packed和MXFP8。
per-tensor对FP32旋转结果直接量化；V4尾64维保存BF16，前448维group64量化；
MX先将旋转结果舍入BF16，再对全部512维group32量化。packed页内data与scale分区不变。

每个program处理B个token，B与W=num_warps联合调优；海光硬件wave64，
编译器根据张量布局分配lane/warp，不能将W等同于token数。
YAML配置名`fused_deepseek_v4_kv_rope_insert`，key包括M、缓存格式、页大小与真实页/token stride。
`grid=ceil(M/B)`；B1负slot可标量提前退出，其余路径在position/KV/表读取前mask。
当前FlagTree默认`FLAGTREE_AABS=1`，在调优时将超过M的B缩为`next_power_of_2(M)`、保留W。
因此YAML六组是初始候选，实际还有B1/W2、B1/W4、B2/W2、B2/W4、B4/W4；已额外逐项验证。
例如M1的B1/W4可以来自B8/W4或B16/W4缩块，最终执行配置以调优记录为准。

分组UE8M0从amax的FP32指数/尾数直接计算，构造精确的倒数二次幂，省去除448、log2、ceil、exp2。
前提为有限BF16来源、amax的1e-4下限及支持的有限RoPE结果。FP8符号、RNE中点和scale边界严格比较。
RoPE显式使用两项cos FMA，另一乘积独立舍入FP32，匹配固定海光官方编译结果；关闭隐式融合。
最终采用的轻量E4M3路径复用海光已有`_f32_to_fp8_e4m3fn`：normal值用位域重定偏置和ties-to-even尾数舍入，
subnormal用magic-add舍入到2^-9格点，显式保留符号。分组量化已保证范围，per-tensor另保留饱和限幅。

融合、分块、位域scale是通用优化方法；wave64候选选择、实际编译器数值路径和OCP FP8编码
需要gfx936实测。不能把所有收益归因于某一个优化点。

## 支持范围

连续BF16 KV[N,512]，INT64 slots[M]/positions[N]，FP32 cos/sin[P,64]，M<=N。
有效slot须唯一且在缓存内，有效position合法，per-tensor scale为正有限FP32单元素。
输入与缓存互不别名；Host仅检查元数据，不同步读取GPU数据检查数值或索引。
普通缓存支持不重叠的真实token/page stride；packed token stride须等于记录宽度，页可有padding。
V4页与storage offset须BF16对齐。非法元数据/不支持布局抛异常。

M0在元数据检查后返回；负slot不读取相应position、KV、旋转表，允许全负slot配空表。
本轮针对有限输入及保持有限的相关中间结果，不承诺任意NaN/Inf量化、别名或重复slot行为、训练反向。

## 官方基线与计时

V4性能基线使用固定LightOp e03b8e8的无KVNorm零Q HIP入口，官方计算CU未修改，
仅加私有注册并通过标准HIPIFY编译。库SHA256锁定为
`59ec2cf5b8e29d7c6bfc8a4b8d633c5cd65b0bcdb988e2bb9a3c0095e4d63cf3`。
设置`KV_ROPE_HYGON_NATIVE_LIBRARY`指向验证过的私有库；缺失或哈希不符时明确报错。
适配负slot的比较/where和零Q构造包含在完整接口计时内，官方原生kernel负责全部数值计算和写缓存。
实测此包装有compare、scalar fill、where、native共4个设备kernel；SUT为1个。
另记录有效position输入下的裸原生三点诊断，避免把全部加速归因于数值kernel。
其余三格式没有匹配官方性能基线，只使用独立CPU参考做功能检查。

沿用仓内`base.Benchmark`，四格式功能测试，V4十点标准shape做kernel和cudagraph计时，
seed0、零cache、M>1末slot=-1，warmup1000ms、rep100ms、parallel0。
加速比=官方完整接口/SUT；报告逐点和等权算术均值，几何平均补充，逐点0.9门槛。
编译、首次调优和生成输入不算稳定执行延迟。

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

## BW1000实测结果（2026-11-23）

最终保留轻量编码trial1：冷编译四格式4/4；完整功能110通过、8项MUSA专用跳过；CPU参考43通过。
38项布局/边界、24项初始候选随机、96项候选负零、200项AABS派生配置、40份实际输入均通过。
包括63,488个有限FP16编码值和32,640个非负有限BF16组最大值的scale/payload检查。
pre-commit及CI helper54项/CI pin检查通过；公共API与实际设备kernel均已核对。

| V4 packed正式十点 | 算术平均加速比 | 几何平均 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| Kernel | 3.291× | 3.269× | 2.803× | 4.096× |
| Graph | 5.914× | 5.750× | 3.272× | 7.405× |

两模式各十点均>=0.9，未触发强制deep-opt。Kernel使用空闲GPU3，Graph使用空闲GPU1，均在各自卡上
对基线/SUT成对计时；共享环境保持默认频率。不同模式或跨卡trial的差异不能作为单个优化的精确因果收益。
另有同GPU0三点Graph诊断：裸native/SUT约4.458×、3.296×、2.902×，是补充采样，不能当十点均值。

轻量编码与首版对比使用相同六组初始候选、AABS设置、scale和RoPE公式及计时方案。
同B4/W1、M16384的生成汇编静态指令1278->636，VGPR75->74，LDS/private均0；
这是静态代码证据，不能当动态执行指令计数或显著寄存器下降。GPU0三点SUT短Graph较首版约快1.13/1.44/1.49倍。
正式首版Kernel/Graph均值3.055/4.694倍，完整快照留档；最终保留trial1，不再扩大搜索。

本阶段完成四格式功能与V4 packed性能验收，其余三格式仍缺匹配官方性能基线；未运行模型端到端。
先选好空闲卡再导入`flaggems_vllm`：镜像中的flash-attn可能在导入时初始化设备，默认无可见卡时会报错。
