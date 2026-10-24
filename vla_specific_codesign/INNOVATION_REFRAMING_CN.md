# 强化后的创新点：Exact Operator–Pipeline Co-Design for Flow-based VLA

## 一、建议的论文主线

不再将工作描述为三个松散的优化：

> 流水线 + 普通算子融合 + 跨帧重用

而应收敛为：

> **现有flow-based VLA runtime将Prefix构建和迭代Action生成视为两个粗粒度阶段，既引入全局同步，又切断了Prefix KV生产者和第一次Action Attention消费者之间的直接数据通路。我们重新暴露当前帧内部的精确依赖，并联合设计流式KV生产—消费、层级wavefront以及重复去噪边界kernel，在不使用stale representation、不减少去噪步数的条件下缩短关键路径。**

推荐题目方向：

> **Exact Operator–Pipeline Co-Design for Flow-based VLA Inference**

跨帧数据重用降为可选扩展，不再作为严格等价主线的核心贡献。

---

## 二、核心贡献一：Exact Prefix–Action Wavefront

### 问题

传统执行先完成全部Prefix layer并生成完整KV cache，随后才执行第一次Action
Expert。它隐含了一个并非模型依赖所要求的全局屏障。

### 方法

对每层建立三类依赖：

\[
P_l \leftarrow P_{l-1},\qquad
A_l \leftarrow A_{l-1},\qquad
A_l \leftarrow KV(P_l)
\]

对应合法调度：

```text
P0
P1 || A0
P2 || A1
...
P17 || A16
A17
```

该调度只重排同一请求、同一当前帧的精确计算，不使用上一帧KV。运行时以
Prefix的compute work和Action Expert的memory work互补为依据决定是否共运行，
而不是假设CUDA stream天然无冲突。

### 当前实验

- H100：单独减少0.129654 ms，端到端下降2.0766%；
- RTX 4090：单独减少0.355203 ms，端到端下降1.1400%；
- H100理想资源上界最多隐藏0.162067 ms，说明默认0.8共运行效率没有超过依赖/资源上界。

### 与已有异步VLA的区别

现有跨帧异步方法通常用上一帧Prefix或旧KV先启动Action Expert；本方案只使用
当前请求中已经完成的layer/tile，因此目标是数值等价的intra-request pipeline。

---

## 三、核心贡献二：Streaming Prefix-KV Producer–Consumer

### 问题

传统第一次Action Attention需要经历：

```text
Prefix KV生成
→ 写入HBM KV cache
→ 等待完整Prefill
→ 第一次AE从HBM读取
```

### 方法

Prefix QKV kernel产生一个K/V tile后形成双路输出：

```text
Prefix K/V tile
├── 写入KV cache，供后续9次AE读取
└── 直接更新第一次AE的online-attention accumulator
```

在线softmax只是实现该流式消费者的机制，不作为独立创新声明。

更重要的是，Action layer \(A_l\) 的ready time从“Prefix layer \(P_l\)全部结束”
提前为“\(P_l\)的QKV/KV tile ready”。

### 当前实验

- 第一次AE直接消费14.0625 MB Prefix K/V；
- H100 HBM→片上资源时间减少0.003336 ms；
- RTX 4090减少0.012861 ms；
- 单独减少流量不改变关键路径，因为Prefix Attention本身compute-bound；
- 与早释放wavefront结合后，H100收益从2.0766%提升到2.1988%，RTX 4090从1.1400%提升到1.2071%。

这个“单独流量下降但延迟不变”的结果应如实报告。它证明优化必须与ready-time
pipeline联合设计，也恰好支撑operator–pipeline co-design的论文主线。

---

## 四、核心贡献三：Repeated Denoise-Boundary Fusion

### 问题

flow-matching每步存在VLA特有边界：

\[
H_t \xrightarrow{W_{out}} v_t
\rightarrow x_t+\Delta t v_t
\xrightarrow{W_{in}} E_{t+\Delta t}
\]

分离执行会物化velocity和updated action，并在10次去噪中重复产生短kernel。

### 方法

融合：

```text
Action OutProj + Euler Update + Next-step Action InProj
```

保持投影和Euler FLOPs不变；第一次input projection及最后一次output+Euler仍按
真实边界单独处理。

### 当前实验

- 边界kernel由30个降为11个，删除19次launch；
- 每次完整推理减少0.1092 MB边界物化；
- 假设3 μs启动开销时，H100节省0.057032 ms，RTX 4090节省0.057106 ms；
- launch=0时只剩约0.00003–0.00011 ms流量收益，说明该优化的主要价值确实来自小kernel调度，而不是虚构大规模HBM节省。

这项贡献具有flow-matching VLA辨识度，但最终必须用真实kernel launch测量校准。

---

## 五、Supporting Action Expert Fusion

QKV→online attention和gated-MLP fusion继续保留，但定位为支撑技术：

- H100 Action Expert节省0.123184 ms；
- RTX 4090节省0.585055 ms；
- H100消除369.14 MB HBM中间流量；
- FLOPs保持不变。

论文中不应把普通QKV/MLP融合单独写成“首次提出”，而应说明它们用于形成更适合
wavefront和去噪runtime的Action kernel suite。

---

## 六、完整共设计效果

默认参数：18层Prefix/Action、10步去噪、overlap efficiency=0.8、kernel
launch=3 μs。

| GPU | 扩展基线 | 完整共设计 | 加速 | 延迟下降 |
|---|---:|---:|---:|---:|
| H100 BF16 | 6.243494 ms | 5.935185 ms | 1.051946× | 4.9381% |
| RTX 4090 BF16 | 31.157441 ms | 30.183415 ms | 1.032270× | 3.1261% |

扩展基线公平地加入原GenZ图中缺失的action input/output projection与Euler
Update；融合方案没有使用“原基线不计开销、优化后突然消失”的不公平口径。

### 不依赖launch假设的结果

将kernel launch设为0：

- H100仍有1.042579× / 4.0840%解析收益；
- RTX 4090仍有1.030415× / 2.9517%解析收益。

因此完整收益并非完全由假设的3 μs启动时间产生。

### 去噪步数趋势

H100完整共设计延迟下降从4步的4.05%增长到20步的5.80%；RTX 4090从2.19%
增长到4.21%。这与“Action kernel优化会被重复N次，而Prefix wavefront只覆盖
第一次AE”的理论判断一致。

---

## 七、建议直接写进论文的贡献表述

1. **Fine-grained exact VLA dependency graph.** We expose layer- and tile-level
   producer–consumer dependencies between current-frame prefix construction and
   the first Action-Expert pass, removing a runtime-induced global barrier
   without stale representations.

2. **Streaming Prefix-KV producer–consumer execution.** We co-design Prefix KV
   generation, first-pass online Action Attention, and mandatory KV-cache
   materialization, allowing each Action layer to start at KV-tile readiness
   while retaining the cache for later denoising steps.

3. **Flow-matching-aware repeated Action kernels.** We fuse the VLA-specific
   output-projection–Euler–next-input boundary and integrate it with a
   resource-aware Action kernel suite whose benefits scale with denoising count.

---

## 八、目前可以说与不能说的结论

### 可以说

- 已实现依赖显式、流量守恒、FLOPs守恒的扩展Roofline原型；
- 所有主方案不使用stale KV、不减少denoise step；
- 分析结果显示H100 4.94%、RTX 4090 3.13%的模拟延迟下降；
- 5项自动化测试验证层数、依赖、后续去噪不变、FLOPs守恒和组合边界；
- 收益随去噪步数和共运行效率呈符合理论的趋势。

### 还不能说

- 不能称这些数字为真实GPU加速；
- 不能声称online softmax、普通QKV或MLP fusion是新算法；
- 不能声称AdaRMSNorm/RoPE已经获得数值收益，因为当前完整DAG尚未建模；
- 不能在没有OpenPI数值对齐前断言bitwise equivalent；
- 不能在没有Nsight并发profile前把overlap efficiency=0.8当成硬件事实。

下一阶段最重要的工作是实现Prefix QKV→Action Attention原型kernel和层级runtime，
用Nsight测量co-run slowdown、KV直供流量以及数值误差，再用真实数据替换当前显式参数。
