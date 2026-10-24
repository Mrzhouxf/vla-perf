# VLA-Perf 原理拆解笔记

> 论文：*How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf* (Jiang et al., 2026, arXiv:2602.18397)
> 仓库：`/home/zxf/vla-perf`，核心代码 `vla-perf/pi0_perf.py` + `vla-perf/perf_utils.py` + `genz/` 后端

VLA-Perf 是一个**解析式（analytical）性能建模工具**，不跑真实模型，而是用 roofline 思路根据模型结构 + 硬件规格 + 网络条件，估算 VLA 模型推理延迟。它建立在 GenZ LLM Analyzer 之上，并为其扩展了"并行解码"建模（用于扩散动作头）。

---

## 一、VLA 模型每个阶段的作用与意义

以论文主研究对象 **Pi0**（Physical Intelligence）为例，它是一条三阶段流水线：

```
图像 + 语言指令 ──► [1. Vision Encoder] ──► 视觉token ──► [2. VLM Backbone] ──► 语义特征/KV cache ──► [3. Action Expert] ──► 连续动作
```

### 阶段 1：Vision Encoder（视觉编码器）—— Pi0 用 SigLIP SoViT-400m
- **作用**：把多路相机图像编码成视觉 token 序列，供后续 VLM 消费。
- **意义**：是 VLA "看"的能力来源。多视角（Pi0 默认 3 路相机）让模型获得空间感知。它是 encoder-only ViT，**一次性处理所有 patch（prefill 语义）**，没有自回归。
- **配置**（`vla_models.py:73`）：hidden=1152, layers=27, heads=16, 256 token/图。

### 阶段 2：VLM Backbone（视觉-语言主干）—— Pi0 用 Gemma 2B
- **作用**：把视觉 token 和语言指令 token 一起做 prefill，融合多模态信息，产出供动作专家查询的 KV cache。
- **意义**：是 VLA "理解"的能力来源——把"看到什么"和"被告知做什么"对齐到统一表示空间。其输出（KV cache）成为动作专家的条件输入。
- **配置**（`vla_models.py:185`）：hidden=2048, layers=18, heads=8, **1 KV head（MQA）**, head_dim=256。序列长度 = `256×3 视觉 + 32 语言 = 800` token（`pi0_perf.py:92-95`）。

### 阶段 3：Action Expert（动作专家）—— Pi0 用 ~300M Diffusion Transformer (DiT)
- **作用**：以 VLM 的 KV cache 为条件，通过 **Flow Matching（流匹配）** 生成连续动作。每生成一组动作（action chunk）需 **N 次去噪迭代**（默认 10 步，`pi0_perf.py:97`），每次迭代是 DiT 的一次前向传播。
- **意义**：是 VLA "做"的能力来源，也是 **Pi0 相对 OpenVLA 的关键架构差异**——OpenVLA 用 7B Llama 自回归逐 token 离散化动作（慢、串行），而 Pi0 用小 DiT 并行解码 + 去噪（快、可一次性出整段动作轨迹）。
- **配置**（`vla_models.py:207`）：hidden=1024, layers=18, heads=8, 1 KV head, head_dim=256。action_chunk_size=50（一次预测 50 个未来动作），DoF=14（双臂 7+7）。

> **对比 OpenVLA**（`openvla_perf.py`）：双视觉编码器（DINOv2+SigLIP）→ Llama2-7B → 自回归解码 7 个离散动作 token。阶段 3 用的是 `collect_decode_perf`（串行 decode），而非扩散。

---

## 二、每阶段的细节执行过程 & 论文分析过程

### 建模总框架（三个 collect 函数，`perf_utils.py`）

所有阶段最终都落到 GenZ 的三类建模调用之一：

| collect 函数 | GenZ 后端 | 用途 | 对应阶段 |
|---|---|---|---|
| `collect_prefill_perf` (`perf_utils.py:421`) | `prefill_moddeling` | 一次性处理所有输入 token | 视觉编码、VLM |
| `collect_decode_perf` (`perf_utils.py:520`) | `decode_moddeling` | 逐 token 自回归解码 | OpenVLA 动作、自回归对照 |
| `collect_parallel_decode_perf` (`perf_utils.py:615`) | `parallel_decode_modeling` | 多 token 并行解码 + 对 KV cache 的 cross-attention | **Pi0 扩散动作头（核心新贡献）** |

每个 collect 函数都做同样三件事：
1. **遍历并行策略**：`get_parallelism(num_devices)` 枚举所有合法的 (tensor_parallel, pipeline_parallel) 组合（`perf_utils.py:278`）。
2. **扫描 batch size**：从 1 开始 ×2 倍增直到上限，对每个 (tp, pp, batch) 调一次 GenZ 建模。
3. **后处理**：用 `evaluate_boundness` 算出该配置是计算/内存/通信 bound，以及操作强度（op intensity）；再用 `get_optimal_df`+`get_pareto_df` 只保留 Pareto 最优的 batch 配置。

### GenZ 后端的 roofline 内核（`genz/GenZ/operator_base.py`）

对每一层算子，延迟由三项取最大值决定（roofline，**取 max 不是求和**，`operator_base.py:289`）：
- **计算时间** = `effective_num_ops × 2 / op_per_sec`（`operator_base.py:108, 273,277`）。`×2` 因为每个 MAC = 1乘+1加；`op_per_sec` 来自硬件在该精度下的 TFLOPS（经 `bit_multiplier` 折算精度）。
- **内存时间** = `Σ tensor_size × bit_multiplier / bandwidth`，分片上/片外（`operator_base.py:95-102, 148-161`）—— 带宽来自硬件显存带宽；每个 tensor 的 residency（on-chip/off-chip）决定用哪条带宽。
- **通信时间**：多卡时 AllReduce/AllGather/All2All 的集合通信开销（`operator_base.py:163`，`collective_times.py`）。

`exec_time = max(compute_time, memory_time, comm_time)`（`operator_base.py:289`）；boundness 判定 = compute/memory 比值（`:291-295`）。逐层算子 `exec_time` 累加得单步总延迟（`analyse_model.py:87-95` 的 `get_summary_table`）。`evaluate_boundness`（`perf_utils.py:87`）再对整模型把三类总时间比较，判定 Comp/Mem/Comm bound，并算加权平均 op intensity = 总 FLOPs / 总内存字节。

**算子 FLOPs 公式**（`operators.py:146-214`）：GEMM = `B·M·N·K`；Logit(QKᵀ) = `B·H·M·N·D`；Attend(softmax·V) = `B·H·M·N·D`。这天然区分了 compute-bound（大 M·N·K 的 prefill GEMM）与 memory-bound（M=1 的串行 decode）。

---

### 阶段 1 细节：视觉编码器建模

**执行过程**（`get_pi0_vision_perf`, `pi0_perf.py:145`）：
- 调 `collect_prefill_perf`，`input_tokens=256`（每图 256 视觉 token）。
- **关键技巧**：`batch_size_multiplier=config.vision_frames`（=3，`pi0_perf.py:166`）—— 把 3 路相机当成 batch 维度，一次 prefill 同时处理 3 路视角，等效 batch=3 的视觉前向。
- 自动选精度：`get_best_precision_for_system`（`perf_utils.py:145`）——硬件不支持 bf16 就降级到同族 fp16 等。
- 输出：vision 延迟 + boundness + 权重/KV 内存。

**论文分析**：把视觉当作纯 prefill（encoder-only，无 KV cache 复用、无 decode），强调它相对整体是轻量开销，但随相机路数线性增长（见实验 3 长上下文）。

### 阶段 2 细节：VLM 主干建模

**执行过程**（`get_pi0_vlm_perf`, `pi0_perf.py:178`）：
- 调 `collect_prefill_perf`，`input_tokens=config.vlm_sequence_length` = 800（256×3 + 32）。
- 一次性 prefill 全部视觉+语言 token，建立 KV cache。
- 同样扫并行策略与 batch。

**论文分析**：VLM 是 prefill 主导（大序列一次性灌入），其 KV cache 成为阶段 3 的输入条件。论文关注 VLM 规模放大对延迟的影响（实验 2：把 Gemma 2B 换成 4B 即 Pi0.6）。

### 阶段 3 细节：动作专家（扩散 / Flow Matching）建模 —— 论文核心创新

**执行过程**（`get_pi0_action_expert_perf`, `pi0_perf.py:211`）：
1. 调 `collect_parallel_decode_perf`，参数：
   - `input_tokens = vlm_sequence_length`（=800）：作为 **VLM KV cache 的上下文长度**，动作 token 对它做 cross-attention。
   - `output_tokens_parallel = action_chunk_size`（=50）：50 个动作 token **同时并行解码**。
   - `self_attention = True`（`pi0_perf.py:264`）：扩散模型中并行动作 token 之间互相 attend（DiT 的 self-attention）。
2. **乘以去噪步数**：`r["time_ms"] = r["time_ms"] × denoising_steps`（`pi0_perf.py:272`）。单步并行解码延迟 × N 次去噪 = 动作专家总延迟。
3. 这就把 Flow Matching 建模为"N 次加宽的 decode 前向"。

**parallel_decode 后端**（`genz/GenZ/LLM_inference/llm_parallel_decode.py:31`）：
- `create_full_parallel_decode_model`（`get_language_model.py:344-432`）构造一个"并行解码"模型视图：每层 = `mha_flash_attention_parallel_decode` + `ffn_prefill`。
- **FFN 用 prefill 版**（`get_language_model.py:409`，注释 "process multiple tokens in parallel"）：FFN 的 GEMM 用 `M = output_gen_tokens_parallel`（=50），而非 decode 的 M=1，所以 FFN 足够大、compute-bound。
- **去噪步数乘子在 GenZ 之外施加**（`pi0_perf.py:272`）：GenZ 只算单步，Pi0 再 `× denoising_steps`。前提假设：每步去噪成本相同——因为动作专家每步都 cross-attend **同一份** VLM KV cache，KV cache 不随步数增长。

#### cross-attention + self-attention 的算子级建模（`attention.py:101-184`，论文核心新贡献）

每层 `mha_flash_attention_parallel_decode` 发射如下算子（`P = action_chunk_size`，`L = vlm_sequence_length`，`H = heads`，`Hkv = kv_heads`，`Dq = head_dim`）：

```
QKV        : GEMM [B,P,D]×[(H·Dq+2·Hkv·Dq),D] → [B,P,QKV]          AC_onchip
Logit Pre  : QKᵀ  [B,H,P,L,Dq,Hkv] → [B,H,P,L]                     AC_onchip   ← 对 VLM KV cache 的 cross-attn
Attend Pre : softmax·V [B,H,P,L,Dq,Hkv] → [B,H,P,Dq]              AC_onchip
[若 self_attention 且 P>1:]
QKV Self   : GEMM [B,P,D] → [B,P,QKV]                              AC_onchip
Logit Self : [B,H,P,P,Dq,Hkv]                                      AC_onchip   ← 动作 token 间 P×P self-attn
Attend Self: [B,H,P,P,Dq,Hkv] → [B,H,P,Dq]                        AC_onchip
Out Proj   : GEMM [B,P,H·Dq]×[H·Dq,D] → [B,P,D]                   AC_onchip
[若 tp>1:] AllReduce over [P,D]
```

**三个关键建模决策（这是论文相对普通 prefill/decode 的创新点）**：
1. **多 token "query batch"（P 而非 1）**：不同于 decode 固定 M=1，parallel decode 固定 M=P，使 FFN/QKV GEMM 大到 compute-bound，匹配扩散 DiT 的计算特征。
2. **显式对冻结 KV cache 的 cross-attention，且 K/V 标记 `AC_onchip`**：`Logit Pre`/`Attend Pre` 用 `L = input_sequence_length` 作 K/V 维度，但 K/V 的 residency 设为片上——建模"VLM KV cache 已驻留、每个去噪步不重读 HBM"。这正是扩散循环便宜的根因，也使 `time × N_steps` 线性缩放成立。
3. **可选 chunk 内 self-attention**：`self_attention=True`（Pi0 动作专家，`pi0_perf.py:264`）时加第二组 QKV 投影 + P×P 注意力，让一段动作轨迹的 token 在同一去噪步内互相影响（flow-matching DiT 的 self-attn）。

**KV cache 处理小结**：

| 阶段 | 是否建模 KV cache | 方式 |
|---|---|---|
| Vision | 否 | prefill 256 token，encoder-only，不留下游 KV |
| VLM | 是（产出） | prefill summary 的 `KV Cache (MB)`，大小 ≈ `2×18×800×1×256×2 ≈ 14.8 MB`（`analyse_model.py:75-76`） |
| Action Expert | 是（消费，片上） | cross-attn 的 K/V 用 `AC_onchip`，每去噪步不重读 HBM |

**论文分析（实验 4、5）**：
- 实验 4 扫 (去噪步数 × action chunk) 二维网格，展示两者对延迟的影响与权衡——chunk 越大单步越慢但摊销到每动作越便宜；步数越多越准但线性增延迟。
- 实验 5 直接对比 **扩散 vs 自回归**：自回归动作时间 = `decode_单步 × DoF × chunk_size`（`pi0_perf.py:1840`，逐 token 串行）；扩散 = `并行单步 × denoising_steps`。结论：扩散随 chunk 增长**远好于**自回归（自回归是 O(chunk)，扩散单步是并行 O(1) per step），这正是 README 图 `pi0_autoregressive_vs_diffusion_chunk_size_B100.png` 的论点。

---

## 三、端到端组装与部署场景建模

### E2E 延迟（`get_pi0_e2e_perf`, `pi0_perf.py:377`）
`E2E = vision + vlm + action_expert`，三阶段串行**纯求和**（`pi0_perf.py:578-582`，Pi0 无 VLM 自回归解码、无阶段间 overlap 建模）。先 `check_model_fits_system`（`pi0_perf.py:284`）判断模型权重+KV cache 是否放得下显存（按精度算每参数字节数，`pi0_perf.py:322`），放不下则跳过该配置。

**batch 对齐**：三阶段各自 Pareto 最优的 batch 可能不同（如视觉只到 batch=4，动作到 16）。`pad_batch_sizes`（`pi0_perf.py:501-567`）对缺失 batch 用**线性缩放**补齐：`scaled_time = base_time × target_batch / base_batch`。Pareto 过滤（`get_optimal_df`, `perf_utils.py:386-418`）只保留"batch 翻倍带来 >2× 收益"的配置。

### 网络延迟模型（`network_latency.py`）——纯解析，不依赖 GenZ
**核心公式**（`calculate_transfer_time_ms`, `network_latency.py:442`；`estimate_image_latency`, `:468`）：
```
transfer_time_ms = (data_bytes × 8 × protocol_overhead) / (bandwidth_mbps × 1000)
total_latency_ms = base_latency_ms(传播+处理) + transfer_time_ms
```
- `protocol_overhead = 1.05`（5% TCP/IP 头开销）。
- 区分 **upload/download** 带宽与 **效率因子**（如 5G 上行 320Mbps × 0.25 = 80Mbps 实际）。
- 三类数据：图像（Robot→Server 上传）、动作（Server→Robot 下载）、KV cache（分布式推理时下发）。
- 网络：4G/5G、WiFi 5/6/6E/7、以太网 1G–400G、云链路（fast/slow），各自有带宽+base_latency 配置（`network_latency.py:86-215`）。

### 部署场景（实验 6、7）
- **实验 6 设备 vs 服务器**（`run_device_vs_server_comparison`）：对比 on-device（Jetson Thor，无网络延迟）vs edge-server / cloud（需加图像上传+动作下发的网络延迟）。E2E = compute + network。
- **实验 7 设备-服务器协同**（`run_device_server_collaboration_comparison`，Helix 式 split inference）：**视觉编码放设备端**（图像不上传，只传视觉 token / KV cache 到服务器），VLM+动作放服务器。研究"切分在哪一刀"能省网络开销。

---

## 四、7 个实验全景（`pi0_perf.py`）

| # | 函数 | 扫什么 | 回答什么 |
|---|---|---|---|
| 1 | `get_all_pi0_perf` | 默认配置跨硬件（A100/H100/B100/4090/Jetson Thor × 1/2/4 卡 × bf16 × 10 去噪步） | Pi0 在各级硬件上多快、各组件 boundness |
| 2 | `get_model_size_scaling_perf` | 放大各组件 | 哪个组件放大最拖延迟 |
| 3 | `run_long_context_experiment` | 历史帧数 [1,10,100,1000,10000] | 长上下文代价（log-log） |
| 4 | `compare_denoising_steps_action_lengths` | 去噪步 [1,5,10,20,50] × chunk [5,25,50,100,250] | 扩散两参数权衡热图 |
| 5 | `compare_autoregressive_vs_diffusion` | 4 种动作生成策略 × chunk × DoF | 哪种动作生成策略快 |
| 6 | `run_device_vs_server_comparison` | 部署位置 × 网络 | on-device 何时能与服务器匹敌 |
| 7 | `run_device_server_collaboration_comparison` | split 切分点 × 全网络 | 设备-服务器协同是否更优 |

**实验 2 模型变体**（`pi0_perf.py:800-1021`，`create_action_expert_config_from_vlm` 构造）：
- **pi-0**：SigLIP-So400m + Gemma-2B + 小动作专家
- **pi-0-L**：SigLIP-giant(1B) + Llama2-7B + 动作专家
- **pi-0-XL**：SigLIP-giant + Llama2-13B + 动作专家
- **pi-0-XXL**：SigLIP-giant + Llama2-70B + 动作专家
> 单卡、固定 10 去噪步。看参数量与各组件/总延迟随规模怎么长。

**实验 3 长上下文增量机制**（`pi0_perf.py:1227-1415`）：每时间步 = 3 帧 × 256 = 768 token。
- t=1：标准 prefill（视觉 + VLM prefill 768+32 + 动作专家对 800 KV）。
- t>1：VLM KV cache 已持有 `(t-1)·768+32` token，新帧通过 **VLM 的 parallel_decode** 增量加入（`input_tokens=(t-1)·768+32`, `output_tokens_parallel=768`，`pi0_perf.py:1305-1314`）；动作专家的 cross-attn 上下文 `L = t·768+32` 随历史线性增长。报告每阶段延迟 + VLM KV 内存。

**实验 5 的 4 种 Setup**（`pi0_perf.py:1739-2108`）：
- **A 自回归 VLA**：用 VLM 主干当动作预测器，`collect_decode_perf` 串行解码；`action_time = decode_单步 × DoF × chunk`（`:1840`）。
- **B 小扩散**：真实 Pi0 动作专家（300M），parallel decode × steps。
- **C 大扩散**：用 VLM 规模的模型当 DiT，`collect_parallel_decode_perf` 并行 `DoF×chunk` token × steps（`:1959-1974`）。
- **D 并行自回归**：VLM 主干 `collect_parallel_decode_perf` 一次并行出全部动作 token（无去噪）。
> 扫 chunk [1,5,10,50,100] 与 DoF [7,14,21,28,35,42]，论证扩散(B/C) 何时胜自回归(A/D)。

**实验 6 部署类别**（`pi0_perf.py:3086-3371`，batch=1 单卡）：
1. **on-device**：Jetson Thor / 4090 / B100，无网络。
2. **edge-server**：4090 / B100，网络走 以太网1G/10G、WiFi6/7、5G/4G。
3. **cloud**：B100 + 两跳网络对：`10G以太网+Fast云`、`4G+Slow云`。
> E2E = compute + 图像上传 + 动作下载。`freq_async_hz = min(inference_hz, network_hz)`（`:3257,3348`）。

**实验 7 三种场景**（`pi0_perf.py:3505-3897`，server=B100, device=Jetson Thor）：
1. **device only**：三阶段全在 Jetson。
2. **server only**：三阶段全在 B100，图像上传+动作下载。
3. **device-server 协同（Helix 式 split）**：视觉+VLM 在 B100 服务器跑，**VLM 的 KV cache 下载到设备**，动作专家（扩散）在 Jetson 本地跑（`:3764-3895`，KV 下载延迟用 `estimate_kvcache_latency`）。
> 协同 E2E = `图像上传 + 服务器VLM + KV下载 + 设备动作`（`:3800,3855`）；异步吞吐 = `min(server_vlm_hz, network_hz, device_action_hz)`（`:3813,3874`），瓶颈在最慢一环。

每个实验产 CSV 到 `perf_results/`，`plot_scripts/print_*.py` 再读 CSV 出图表与 LaTeX 表。

---

## 五、关键架构常数（`pi0_perf.py:92-98`）

| 常数 | 值 | 含义 |
|---|---|---|
| `PI0_VISION_TOKENS` | 256 | SigLIP 每图输出 token |
| `PI0_VISION_FRAMES` | 3 | 相机路数 |
| `PI0_LANGUAGE_TOKENS` | 32 | 语言指令 token |
| `PI0_VLM_SEQUENCE_LENGTH` | 800 | VLM prefill 序列长 |
| `PI0_ACTION_CHUNK_SIZE` | 50 | 一次预测动作数 |
| `PI0_DEFAULT_DENOISING_STEPS` | 10 | Flow Matching 去噪步 |
| `PI0_ACTION_DOF` | 14 | 双臂自由度 |

---

## 一句话总结
VLA-Perf 把 VLA 推理拆成 vision(prefill) + VLM(prefill) + action(diffusion parallel-decode × N steps) 三段，用 GenZ 的 roofline 解析模型算每段延迟，叠加网络延迟后比较 on-device/edge/cloud/split 四种部署，核心创新是为扩散动作头新增的"并行解码"建模——这让"扩散优于自回归"这一结论有了量化支撑。
