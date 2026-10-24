# VLA overlap-pipeline prototype

此目录独立于 `vla-perf/` 原有实现，不会修改 `pi0_perf.py`、`perf_utils.py` 或 GenZ。当前版本只输出 GenZ 可验证的逐算子数据和依赖安全的端到端基线：

- 使用 GenZ 逐算子的 compute、memory、communication 与有效 roofline latency；
- 不将 `Memory time` 错当作可全部预取的数据；它在 GenZ 中包含输入、权重和输出写回；
- 不假设 SigLIP 可在完整 encoder 完成前输出 VLM 可用的 token；
- 不假设 Action Expert 可早于完整 VLM KV cache 开始。

运行：

```bash
cd /home/zxf/vla-perf
python pipeline_prototype/profile_and_pipeline.py --gpu H100 --bits bf16
```

将生成：

- `results/REPORT.md`：实验报告、阶段瓶颈及依赖安全的端到端基线；
- `operator_breakdown.csv`：全部已展开层的算子细分；
- `category_breakdown.csv`、`latency_type_summary.csv`：按 bound 与计算/访存/通信类型的统计和占比；
- `stage_breakdown.csv`、`pipeline_summary.csv`：阶段和端到端汇总；
- `dependency_safe_operator_schedule.csv`：保持原始依赖顺序的参考时间表；
- `stage_latency_breakdown.png`：可视化。

注意：GenZ 的 compute/memory time 是资源工作量；算子 `Latency` 是两者 roofline 临界路径，不能把二者直接相加。没有一个明确的读/写 DMA、双缓冲容量和 HBM 仲裁模型时，不能再从这些列推导新的访存/计算 overlap 收益。

## Fast-OverlaPIM mapping transformation

`fast_overlapim_mapping.py` 默认输出依赖安全结果：SigLIP 完整输出后 VLM 才开始，因此不存在可合法转换的跨阶段 data spaces。只有在已验证“视觉 encoder 可以流式产出 VLM 可用 embedding”且已验证固定 SM/HBM 分区性能时，才可显式开启 analytical mapping experiment。

```bash
python pipeline_prototype/fast_overlapim_mapping.py --profile-dir pipeline_prototype/results

# Only with an independently validated streaming encoder and partition model:
python pipeline_prototype/fast_overlapim_mapping.py --profile-dir pipeline_prototype/results --resources 2 --chunks 3 --assume-streaming-vision --assume-partitioned-gpu
```

由于论文的 Bank/Channel PIM 资源在 GPU 上没有一一对应物，启用的 `resources` 是固定预算的逻辑执行分区，不代表额外 GPU 或真实 CUDA stream 并发能力。
