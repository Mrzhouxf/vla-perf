# VLA-Perf 三轴优化实验

此目录与原始 `vla-perf/`、`genz/`、已有 `pipeline_prototype/` 和
`operator_fusion_prototype/` 完全分离。它读取逐算子 GenZ profile，在同一
Roofline 口径下模拟三类优化：

1. `pipeline.py`：producer tile ready 后立即启动 consumer tile；
2. `fusion.py`：QKV→online attention 与 gate/up→activation→down；
3. `reuse.py`：相似度门控的跨帧视觉 activation/VLM visual-prefix KV 重用。

`run_ablation.py` 会运行单项、两两组合和三项组合。组合场景按
`reuse → fusion → pipeline` 变换同一份逐算子 counters，而不是把三个独立
加速比相乘，因此已被重用的数据不会再次获得完整的融合流量收益。

## 运行

先生成对应 GPU 的基础 profile：

```bash
cd /home/zxf/vla-perf
python pipeline_prototype/profile_and_pipeline.py \
  --gpu H100 --bits bf16 \
  --output pipeline_prototype/results
```

再运行统一消融：

```bash
python -m three_axis_optimization.run_ablation \
  --profile-dir pipeline_prototype/results \
  --gpu H100 --bits bf16 \
  --output three_axis_optimization/results_h100
```

RTX 4090：

```bash
python -m three_axis_optimization.run_ablation \
  --profile-dir pipeline_prototype/results_rtx_4090 \
  --gpu RTX_4090 --bits bf16 \
  --output three_axis_optimization/results_rtx_4090
```

测试：

```bash
python -m unittest three_axis_optimization.test_models -v
```

## 默认实验参数

- pipeline tiles：8；
- 连续帧数：8；
- 帧相似度：0.92；
- reuse gate：0.85；
- 可复用覆盖上限：0.80，因此视觉首层实际复用率为 `0.92 × 0.80`；
- 跨层可信度衰减：0.97；
- VLM visual-prefix 传播系数：0.25；
- 每个后续帧相似度验证开销：0.02 ms；
- Action Expert：默认禁止跨帧重用。

## 输出

- `REPORT_CN.md`：中文实验报告和解释边界；
- `ablation_summary.csv`：8 种单项/组合消融；
- `stage_ablation.csv`：逐阶段首帧、后续帧和平均延迟；
- `pipeline_segments.csv`：每个 ready-time segment 的资源下界和气泡；
- `fusion_candidates.csv`：每个融合链节省的 HBM/片上流量；
- `reuse_operator_details.csv`：逐算子复用率、计算/访存变化；
- `reuse_similarity_sweep.csv`：相似度敏感性；
- `three_axis_ablation.png`：消融和敏感性图。

## 必须注意的边界

这是 analytical what-if experiment，不是实际 CUDA/Triton kernel 计时。
流水线假设 runtime 支持 tile-ready 双缓冲；跨帧重用属于近似计算，必须在
真实数据集上测量 feature error、动作成功率、cache false-hit 和回退率。
当前 GenZ π0 图未显式包含 AdaRMSNorm、RoPE、Residual、激活函数和 Euler
Update，代码不会为这些缺失算子虚构收益。
