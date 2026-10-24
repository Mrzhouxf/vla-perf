# VLA 推理算子融合预实验：背景、建模方法、实现细节与实验分析

## 摘要

Vision-Language-Action（VLA）模型需要在视觉理解、语言条件建模和连续动作生成之间完成端到端推理。以 π0 为代表的 flow-matching VLA 通常包含 Vision Encoder、VLM Prefix Prefill 和迭代式 Action Expert 三个阶段。与普通 LLM 不同，Action Expert 需要针对同一动作块执行多次去噪，因此短序列算子、权重搬运和中间张量物化会被重复放大。

本工作在不修改原始 VLA-Perf 和 GenZ 的前提下，实现了一个独立的、面向 VLA 算子融合的分析原型。原型首先展开 Transformer repeat 和 Action Expert 的 10 次去噪迭代，获得逐层、逐算子的 compute、memory、communication 和有效 Roofline latency；随后识别相邻 producer-consumer 算子，并通过消除可证明的中间张量访问，重新计算融合链的 Roofline 临界路径。

在 H100/bf16 配置下，组合使用 QKV-Attention 融合与 MLP 中间激活融合，可将模拟端到端延迟从 6.1527 ms 降至 5.9835 ms，对应 1.0283× 加速；其中 Action Expert 从 2.1420 ms 降至 2.0188 ms，对应 1.0610× 加速。在 RTX 4090 上，Action Expert 从 7.2544 ms 降至 6.6694 ms，对应 1.0877× 加速；端到端加速为 1.0207×。实验表明，VLM Prefill 主要受计算限制，单纯减少中间访存难以改变其临界路径；Action Expert 则具有明显的 memory-bound 特征，是当前最值得进行融合优化的阶段。

需要强调的是，本工作是 traffic-aware analytical what-if experiment，而不是实际 CUDA/Triton kernel 测量。模型严格保持 FLOPs 不变，只扣除已经识别并匹配的中间张量访问。AdaRMSNorm、RoPE、Euler Update 等当前 GenZ 图中缺失的算子没有被赋予虚构收益。

---

## 1. 研究背景

### 1.1 π0 类 VLA 的推理结构

当前 VLA-Perf 对 π0 推理过程采用如下阶段划分：

```text
多视角图像
    ↓
Vision Encoder（SigLIP）
    ↓
VLM Prefix Prefill（Gemma）
    ↓
VLM KV Cache
    ↓
Action Expert / DiT
    ↓
重复 N 次 flow-matching denoising
    ↓
动作块
```

本实验使用的默认工作负载为：

- 3 个视觉视角；
- 每个视角 256 个视觉 token；
- 32 个语言 token；
- VLM Prefix 总长度为 800 token；
- 动作块长度为 50 token；
- Action Expert 执行 10 次去噪。

Vision Encoder 和 VLM Prefill 都包含典型 Transformer 算子：

```text
QKV Projection
→ Attention Logit
→ Softmax / Attention
→ Output Projection
→ Gate/Up Projection
→ Activation
→ Down Projection
```

Action Expert 也包含类似结构，但它具有三个值得关注的特点：

1. 动作 token 数较少，GEMM 的矩阵规模小于 VLM Prefill；
2. 相同 Action Expert 权重会在多个 denoising step 中重复访问；
3. Flow-matching 推理还存在 timestep conditioning、Euler update 和下一步动作 embedding 等普通 LLM 不具备的计算边界。

因此，VLA 的最佳融合方向不一定是简单复用面向大 batch LLM Prefill 的优化，而应优先关注重复、短序列、低算术强度的 Action Expert。

### 1.2 算子融合为什么可能降低延迟

相邻算子分开执行时，一个中间张量通常经历：

```text
Producer compute
→ 写入 HBM 或片上存储
→ Consumer 再次读取
→ Consumer compute
```

如果 producer 和 consumer 可以在同一个 kernel、persistent kernel 或 tile pipeline 中执行，中间结果可以保留在寄存器、shared memory 或其他片上存储中：

```text
Producer tile
→ register/shared-memory intermediate
→ Consumer tile
```

融合不会减少模型语义所要求的矩阵乘 FLOPs，但可以减少：

- 中间张量的 HBM 写回；
- Consumer 对中间张量的 HBM 读取；
- 部分片上 score tensor 的物化；
- 实际 runtime 中的 kernel launch 和同步开销。

本实验只对前三项中的可量化数据搬运进行建模。由于 GenZ 没有 kernel launch 参数，实验没有凭空假设 launch overhead。

### 1.3 在线 Softmax 与 Attention 融合

用户提出的“矩阵乘后不必先写出完整结果，而是在计算过程中同步维护指数和归一化量”对应 Online Softmax/FlashAttention 的基本数据流。

对于分块 attention score \(S_i\)，可以维护 running maximum \(m_i\)、归一化分母 \(l_i\) 和输出累加器 \(O_i\)：

\[
m_i=\max(m_{i-1},\max(S_i)),
\]

\[
l_i=e^{m_{i-1}-m_i}l_{i-1}
    +\sum_j e^{S_{i,j}-m_i},
\]

\[
O_i=e^{m_{i-1}-m_i}O_{i-1}
    +\sum_j e^{S_{i,j}-m_i}V_{i,j}.
\]

最终再使用 \(O_i/l_i\) 得到归一化 attention 输出。这样不需要将完整的 \(N\times N\) score matrix 写回 HBM。

不过，GenZ 当前使用的正是 `mha_flash_attention_*` 模型：`Logit` 输出被标记为 `C_onchip`，`Attend` 的 score 输入被标记为 `A_onchip`。因此，当前 baseline 已经近似避免了 attention score 的完整 HBM 物化。本工作不会再次将该矩阵的 HBM 读写计为新收益，只计算进一步消除片上 score 写读的理想收益。

---

## 2. 当前 VLA-Perf/GenZ 模型的缺陷

### 2.1 算子图粒度不足

当前 GenZ π0 图主要包含：

- QKV GEMM；
- Logit；
- Attend；
- Output Projection；
- `up+gate`；
- `down`。

但实际 Transformer/VLA runtime 中还存在：

- RMSNorm/AdaRMSNorm；
- RoPE；
- Softmax 的 max、exp、sum 与归一化；
- residual add；
- SiLU/GELU 和 gate multiply；
- timestep-conditioned scale/shift；
- Euler update；
- action input/output projection 边界。

当前 `up+gate` 只通过矩阵维度隐式表达 gate/up projection，激活函数没有单独的延迟行；Softmax 算术也没有独立行。因此当前 profile 无法直接回答 AdaRMSNorm、RoPE、Euler 等算子的真实延迟，也不能精确估计包含这些操作的融合 kernel。

### 2.2 Roofline 模型不能表达真实 kernel 行为

每个 GenZ 算子的有效延迟为：

\[
T_{op}=\max(T_{compute},T_{memory},T_{communication}).
\]

这一模型适合定位算子是 compute-bound 还是 memory-bound，但不能直接表示：

- kernel launch latency；
- GPU occupancy；
- register pressure；
- shared-memory 容量；
- tensor-core tile efficiency；
- L2/cache reuse；
- 多 kernel 并发时的 SM/HBM 争用；
- 融合后因为寄存器增加导致的 occupancy 下降。

因此，本实验得到的是融合数据流在理想 Roofline 条件下的 latency lower bound，而不是实际 GPU 上必然达到的时间。

### 2.3 Memory time 混合了多类访问

GenZ 的 Memory time 是根据输入 activation、权重和输出的存储位置计算的总搬运时间。它不是一个可以整体提前执行的“prefetch time”。尤其是输出写回必须发生在计算产生结果之后。

本工作因此不再使用“将完整 Memory time 与 Compute time 额外重叠”的方法，而是只在明确的 producer-consumer 边上，扣除可以避免的中间张量写和读。

### 2.4 Attention baseline 已包含部分融合

由于 Logit score 已经放在片上，简单声称“融合 Softmax 后消除了完整 attention matrix 的 HBM 访问”会重复计算优化收益。当前实验对 Attention 分成两个层次：

1. `Logit → Attend`：只消除片上 score materialization；
2. `QKV → Logit → Attend`：进一步消除可以直接消费的 Q/K/V 中间 HBM 访问。

### 2.5 缺少实际 VLA runtime 的算子测量

当前结果来自 GenZ analytical simulator，而不是 Nsight Systems、Nsight Compute、PyTorch Profiler 或真实 Triton/CUDA kernel。因此报告中的 latency 应解释为模拟延迟和理论融合潜力。

### 2.6 还没有覆盖 VLA 最有特色的融合边界

附件中建议的以下方案很有价值：

- AdaRMSNorm + QKV + RoPE；
- Output Projection + Euler Update + Next-step Action Embedding；
- Prefix KV Producer + First Action Attention Consumer；
- VLM layer 与 Action Expert layer 的精细 wavefront pipeline。

但这些算子/依赖没有完整出现在当前 GenZ 图中。本工作将它们写入 `unsupported_vla_specific_candidates.csv`，而不是赋予无法验证的数值加速。

---

## 3. 技术设计

### 3.1 总体流程

融合分析流程如下：

```text
GenZ π0 operator DataFrame
        ↓
展开 Transformer Repeat
        ↓
展开 10 次 Action Expert denoising
        ↓
获得逐算子 compute/memory/communication/latency
        ↓
按 stage / denoising iteration / Transformer layer 分组
        ↓
扫描相邻 producer-consumer 链
        ↓
计算可消除的 HBM/片上中间张量字节数
        ↓
重新计算融合链 Roofline latency
        ↓
聚合为阶段和端到端结果
```

实现文件为：

```text
operator_fusion_prototype/analyze_operator_fusion.py
```

它只读取：

```text
pipeline_prototype/results/operator_breakdown.csv
```

不会修改 VLA-Perf、GenZ 或原始模型定义。

### 3.2 Repeat 与 denoising 展开

GenZ 使用 `Repeat/EndRepeat` 表示相同 Transformer block 的多层重复。分析器首先将它们物化为：

```text
repeat0/QKV
repeat0/Logit
...
repeat1/QKV
repeat1/Logit
...
```

Action Expert 单次 forward 包含 18 个 block；π0 默认执行 10 次去噪，因此分析器进一步增加 `stage_invocation=1...10`。

最终 H100 profile 包含 1902 个算子实例数据行，其中每个实例都记录：

- 所属阶段；
- denoising iteration；
- Transformer layer；
- 前一个和后一个相邻算子；
- compute/memory/communication time；
- 输入、权重、输出和总数据量；
- compute/memory utilization。

### 3.3 融合候选一：Online Attention

匹配模式：

```text
Logit → Attend
```

GenZ 已将 `Logit.Output` 和 `Attend.Input_a` 放在片上。因此可消除的数据量为：

\[
B_{saved,onchip}
=B_{Logit.Output}+B_{Attend.InputA}.
\]

该方案不扣除任何 HBM score traffic：

\[
B_{saved,HBM}=0.
\]

它表达的是更细粒度的 online softmax/attention tile 消费，而不是重复实现一次 FlashAttention 的主要 HBM 优化。

### 3.4 融合候选二：QKV + Online Attention

匹配模式：

```text
QKV → Logit → Attend
```

对于 Vision Encoder 和 VLM Prefill，Q、K、V 都来自同一个 QKV projection。理想融合时，QKV 输出不必写入 HBM后再被 Logit/Attend 读取：

\[
B_{saved,HBM}
=B_{QKV.Output}
+\min(B_{QKV.Output},B_{Q-read}+B_{K-read}+B_{V-read}).
\]

对于完整覆盖的 Q/K/V 边，上式接近：

\[
B_{saved,HBM}\approx2B_{QKV.Output}.
\]

Action Expert 的 Prefix Attention 不同：K/V 来自已经构造好的 prefix KV cache，当前 QKV row 与 Prefix Attention 之间只有 Query 是可直接确认的 producer-consumer 边。为了避免高估，分析器只扣除 Query 的写回和读取：

\[
B_{saved,HBM}=2B_Q.
\]

Prefix K/V 的 HBM 访问仍然保留。

对于 `QKV Self → Logit Self → Attend Self`，当前 action token 的 Q/K/V 都来自 QKV Self，因此可以使用完整 Q/K/V producer-consumer 覆盖。

### 3.5 融合候选三：Action/VLM MLP Fusion

匹配模式：

```text
up+gate
→ activation / gate multiply
→ down
```

GenZ 中 activation 没有独立 row，但其数据关系可以从相邻 GEMM 张量尺寸推导。融合模型假设 gate/up projection 以 token/tile 为单位产生中间结果，激活后立即被 down projection 消费。

可消除的 HBM traffic 为：

\[
B_{saved,HBM}
=B_{up+gate.Output}+B_{down.InputA}.
\]

该模型不会删除 up/down GEMM 的权重访问，也不会减少矩阵乘 FLOPs。

### 3.6 融合延迟模型

对于一个包含若干相邻算子的融合链，baseline 延迟为：

\[
T_{baseline,chain}=\sum_i T_i.
\]

计算、访存和通信工作量分别为：

\[
C_{chain}=\sum_i C_i,
\quad
M_{chain}=\sum_i M_i,
\quad
N_{chain}=\sum_i N_i.
\]

根据 GPU HBM 和片上带宽，将节省的字节转换为时间：

\[
\Delta M_{HBM}=B_{saved,HBM}/BW_{HBM},
\]

\[
\Delta M_{onchip}=B_{saved,onchip}/BW_{onchip}.
\]

融合后的 memory work 为：

\[
M'_{chain}=\max(0,
M_{chain}-\Delta M_{HBM}-\Delta M_{onchip}).
\]

融合链的理想 Roofline 延迟为：

\[
T_{fused,chain}
=\max(C_{chain},M'_{chain},N_{chain}).
\]

最终节省：

\[
\Delta T_{chain}
=\max(0,T_{baseline,chain}-T_{fused,chain}).
\]

该公式具有两个重要性质：

1. 融合后延迟不会低于累计计算时间；
2. 即使消除了大量访存，compute-bound 链也可能完全不下降。

第二点正是 VLM Prefill 实验结果为 1.0× 的原因。

### 3.7 组合场景与防止重复计数

实验定义五个场景：

1. `baseline`；
2. `online_attention_only`；
3. `qkv_online_attention`；
4. `mlp_activation_down`；
5. `combined_qkv_attention_mlp`。

`qkv_online_attention` 已包含内部的 `Logit → Attend`，因此组合场景不会再次叠加 `online_attention_only` 的收益。组合场景只选择：

```text
qkv_online_attention
+
mlp_activation_down
```

两类候选在算子索引上互不重叠。

---

## 4. 实验设置

### 4.1 模型配置

| 组件 | 配置 |
|---|---|
| Vision | π0 SigLIP/SoViT，3 × 256 tokens |
| VLM | Gemma 2B-like，18 layers，800-token prefill |
| Action Expert | 18 layers，50 action tokens |
| Denoising | 10 steps |
| Precision | bf16 |

### 4.2 硬件配置

| GPU | bf16 峰值 | HBM bandwidth | 显存 |
|---|---:|---:|---:|
| H100 | 989 TFLOPS | 3350 GB/s | 80 GB |
| RTX 4090 | 165 TFLOPS | 1008 GB/s | 24 GB |

GenZ 默认片上带宽配置为 18000 GB/s。实验使用单 GPU、TP=1、PP=1，不包含通信时间。

### 4.3 输出数据

每种 GPU 产生：

- `operator_instance_breakdown.csv`：每个算子实例；
- `operator_stage_breakdown.csv`：每阶段、每算子类型汇总；
- `adjacent_operator_pairs.csv`：所有相邻算子关系；
- `fusion_candidates_detailed.csv`：每个融合候选；
- `fusion_candidates_summary.csv`：候选聚合；
- `fusion_scenarios.csv`：各场景的阶段和端到端结果；
- `unsupported_vla_specific_candidates.csv`：无法可信量化的候选；
- `operator_fusion_stage_latency.png`：融合前后阶段延迟图。

---

## 5. 基线算子延迟分析

### 5.1 H100：Vision Encoder

Vision Encoder 总延迟为 0.7091 ms。

| 算子 | 延迟（ms） | 阶段占比 | Bound |
|---|---:|---:|---|
| up+gate | 0.2079 | 29.32% | Compute |
| down | 0.2079 | 29.32% | Compute |
| QKV | 0.1669 | 23.54% | Compute |
| Out Proj | 0.0557 | 7.85% | Compute |
| Logit | 0.0354 | 4.99% | Memory |
| Attend | 0.0354 | 4.99% | Memory |

Vision 的线性 projection 主要 compute-bound，但 Logit/Attend 由于 score tensor 较大而 memory-bound。因此 QKV-Attention 融合可以减少 attention 链的局部等待。

### 5.2 H100：VLM Prefill

VLM Prefill 总延迟为 3.3016 ms。

| 算子 | 延迟（ms） | 阶段占比 | Bound |
|---|---:|---:|---|
| up+gate | 1.9542 | 59.19% | Compute |
| down | 0.9771 | 29.60% | Compute |
| QKV | 0.1527 | 4.62% | Compute |
| Out Proj | 0.1221 | 3.70% | Compute |
| Logit | 0.0477 | 1.45% | Compute |
| Attend | 0.0477 | 1.45% | Compute |

VLM 的所有有效算子均 compute-bound，特别是 FFN 占约 88.8%。因此仅减少中间 activation traffic 不会越过 compute lower bound。

### 5.3 H100：Action Expert

10 次 Action Expert 总延迟为 2.1420 ms。

| 算子 | 延迟（ms） | 阶段占比 | Bound |
|---|---:|---:|---|
| up+gate | 0.8857 | 41.35% | Memory |
| down | 0.4454 | 20.79% | Memory |
| QKV | 0.2657 | 12.40% | Memory |
| QKV Self | 0.2657 | 12.40% | Memory |
| Out Proj | 0.2127 | 9.93% | Memory |
| Logit Pre | 0.0298 | 1.39% | Compute |
| Attend Pre | 0.0298 | 1.39% | Compute |

Action Expert 的主要 GEMM 全部 memory-bound。其 compute utilization 约为 17%–18%，说明短序列计算无法充分摊薄权重搬运，是最适合进行数据驻留和融合的阶段。

---

## 6. 融合实验结果

### 6.1 H100 端到端结果

| 场景 | 端到端延迟（ms） | 节省（ms） | 加速 |
|---|---:|---:|---:|
| Baseline | 6.1527 | 0 | 1.0000× |
| Online Attention only | 6.1344 | 0.0183 | 1.0030× |
| QKV + Online Attention | 6.0450 | 0.1077 | 1.0178× |
| MLP Activation + Down | 6.0912 | 0.0615 | 1.0101× |
| Combined | 5.9835 | 0.1692 | 1.0283× |

### 6.2 H100 分阶段组合结果

| 阶段 | Baseline（ms） | Combined（ms） | 节省（ms） | 加速 |
|---|---:|---:|---:|---:|
| Vision Encoder | 0.7091 | 0.6632 | 0.0460 | 1.0693× |
| VLM Prefill | 3.3016 | 3.3016 | 0 | 1.0000× |
| Action Expert | 2.1420 | 2.0188 | 0.1232 | 1.0610× |

H100 上 Action Expert 的 QKV-Attention 与 MLP 两类融合分别节省约 0.0617 ms 和 0.0615 ms，贡献接近。Vision 的收益主要来自 QKV-Attention 链；Vision MLP 和 VLM MLP 即使消除了中间 HBM traffic，也仍然受累计计算时间限制。

### 6.3 H100 数据搬运结果

| 阶段/融合 | 出现次数 | 节省 HBM | 节省片上访问 | 链局部加速 |
|---|---:|---:|---:|---:|
| Action QKV-Attention | 360 | 158.20 MB | 233.46 MB | 1.1150× |
| Action MLP | 180 | 210.94 MB | 0 | 1.0484× |
| Vision QKV-Attention | 27 | 273.38 MB | 324.00 MB | 1.2398× |
| VLM QKV-Attention | 18 | 140.63 MB | 351.56 MB | 1.0000× |
| VLM MLP | 18 | 1350.00 MB | 0 | 1.0000× |

VLM MLP 虽然理论上可以减少 1350 MB 的中间 traffic，但其计算时间高于调整后的 memory time，所以有效延迟不变。这一结果说明“节省字节数多”不等价于“端到端一定加速”。

### 6.4 RTX 4090 结果

| 场景 | 端到端延迟（ms） | 节省（ms） | 加速 |
|---|---:|---:|---:|
| Baseline | 31.0648 | 0 | 1.0000× |
| Online Attention only | 31.0472 | 0.0176 | 1.0006× |
| QKV + Online Attention | 30.6382 | 0.4266 | 1.0139× |
| MLP Activation + Down | 30.8604 | 0.2044 | 1.0066× |
| Combined | 30.4338 | 0.6310 | 1.0207× |

分阶段结果：

| 阶段 | Baseline（ms） | Combined（ms） | 加速 |
|---|---:|---:|---:|
| Vision Encoder | 4.0208 | 3.9749 | 1.0115× |
| VLM Prefill | 19.7895 | 19.7895 | 1.0000× |
| Action Expert | 7.2544 | 6.6694 | 1.0877× |

RTX 4090 的 HBM 带宽低于 H100，因此 Action Expert 的中间 traffic 对延迟影响更明显，阶段加速达到 1.0877×。但 RTX 4090 的 VLM Prefill 占端到端延迟约 63.7%，且 VLM 仍然 compute-bound，因此 Action 的较大局部收益被 VLM 主导的端到端临界路径稀释，最终端到端加速为 1.0207×。

---

## 7. 结果讨论

### 7.1 为什么端到端加速不是很大

融合仅优化能够消除中间 traffic 的局部算子链。H100 上 VLM Prefill 占 53.7%，RTX 4090 上占 63.7%，但 VLM 基本完全 compute-bound。只做 traffic fusion 无法降低这部分的累计 GEMM 时间。

此外，Action Expert 虽然重复 10 次且 memory-bound，但其 HBM traffic 主要仍来自权重：

- up/gate projection 权重；
- down projection 权重；
- QKV 和 output projection 权重。

当前融合只消除了中间 activation，不会消除模型权重访问。因此 6%–9% 的 Action 阶段加速是合理量级，而不是 2×。

### 7.2 Online Attention 单独收益很小的原因

GenZ baseline 已经使用片上 attention score，因此 Online Attention only 只消除片上 score 写读。H100 端到端仅提升约 0.3%，RTX 4090 约 0.06%。

如果实际 runtime 尚未使用 FlashAttention，真实优化空间可能比该模拟更大；但那需要真实 profiler 证明 baseline 确实将完整 score matrix 写入 HBM。本实验不能在没有证据时假设这部分流量存在。

### 7.3 Action Expert 是下一阶段实现重点

Action Expert 具有稳定的 memory-bound 特征，并且融合候选重复出现：

- 18 layers × 10 steps × 2 attention chains = 360 个 QKV-Attention 候选；
- 18 layers × 10 steps = 180 个 MLP 候选。

因此，实际 kernel 实现应优先考虑：

1. `QKV Self + Online Attention`；
2. Action MLP persistent fusion；
3. Prefix Attention 中只保留 Query 的直接消费，同时继续读取 prefix KV；
4. 将 timestep-conditioned AdaRMSNorm 与 QKV projection 合并。

### 7.4 融合可能产生的负面效果

Roofline 结果没有考虑以下风险：

- QKV/MLP 大融合导致寄存器使用上升；
- shared-memory tile 增大；
- occupancy 下降；
- tensor-core tile 形状变差；
- down projection 的跨 intermediate-dimension reduction 难以完全片上保存；
- 不同 GPU 架构的最佳 tile size 不同。

因此，报告中的 fused latency 应视为 traffic-only 理想值。真实 kernel 如果因 occupancy 下降而增加 compute time，实际收益可能低于本报告。

---

## 8. 正确性与守恒验证

实现包含三个自动测试：

1. 融合链 latency 不得低于累计 compute time；
2. 组合场景不得重复计算 `Logit → Attend` 收益；
3. Action Expert 候选数必须与 18 layers × 10 steps 对应。

测试结果：

```text
test_fused_chain_never_drops_compute                  OK
test_combined_scenario_does_not_double_count          OK
test_expected_action_candidate_counts                 OK
```

逐算子 latency 总和与阶段 baseline 完全一致，说明 Repeat/denoising 展开没有改变原始 GenZ 总延迟。所有融合候选均设置 `flops_conserved=True`。

---

## 9. 尚未完成与后续工作

### 9.1 补齐完整 VLA 推理 DAG

首先需要在模型中显式加入：

```text
AdaRMSNorm
→ QKV
→ RoPE
→ Attention
→ Out Projection
→ Residual
→ AdaRMSNorm
→ Gate/Up
→ SiLU/GELU
→ Multiply
→ Down
→ Euler Update
→ Next-step Input Projection
```

只有完整 DAG 才能量化真正 VLA-specific 的融合，而不仅是 Transformer 通用融合。

### 9.2 增加 launch overhead 与 occupancy 模型

需要从真实 GPU profile 获得：

- 每个 kernel 的 launch time；
- achieved occupancy；
- DRAM bytes；
- L2 hit rate；
- tensor-core utilization；
- register/shared-memory 使用量。

随后将真实数据用于校准 analytical model。

### 9.3 实现 Triton/CUDA kernel

推荐实现顺序：

1. Action `QKV Self + Online Attention`；
2. Action `up+gate + activation + down`；
3. AdaRMSNorm + QKV + RoPE；
4. Output Projection + Euler + Next Input；
5. Prefix KV Producer–Consumer fusion。

每个 kernel 都需要同时比较数值误差、latency、HBM traffic 和 occupancy。

### 9.4 Layer-wise Prefix–Action Pipeline

附件提出的 layer-wise VLM Prefix→Action wavefront 具有潜力，但需要联合模型图证明 Action layer \(l\) 只依赖 Prefix layer \(l\) 的 KV，并需要资源共享模型处理同一 GPU 上 VLM compute-bound 与 Action memory-bound 算子的并发。当前工作没有将无资源约束的双引擎 upper bound 当作真实收益。

---

## 10. 结论

本工作完成了一个不修改原始代码的 VLA 算子融合预实验框架。它提供逐算子延迟、相邻算子关系、融合候选、消除的 HBM/片上字节数以及阶段/端到端 Roofline 结果。

实验得到三个主要结论：

1. VLM Prefill 主要 compute-bound，traffic-only fusion 很难改变临界路径；
2. Action Expert 明显 memory-bound，QKV-Attention 与 MLP 中间激活融合可以产生稳定的局部收益；
3. Online Attention 的主要 HBM 优化已存在于当前 GenZ baseline，因此不能重复计算完整 score matrix 的收益。

H100 上组合融合的模拟端到端加速为 1.0283×，Action Expert 阶段为 1.0610×；RTX 4090 上端到端为 1.0207×，Action Expert 阶段为 1.0877×。这些结果表明算子融合方向有效，但若要形成更显著、更具 VLA 特征的优化，需要继续加入 AdaRMSNorm/RoPE、Euler boundary 和 Prefix KV producer-consumer fusion，并通过真实 GPU kernel 和 profiler 校准模型。

