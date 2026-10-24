# VLA operator-fusion prototype

该目录读取 `pipeline_prototype` 生成的逐算子 GenZ 数据，不修改原始 VLA-Perf 或 GenZ。融合模型遵守三条规则：

1. FLOPs 不减少；
2. 只消除已经匹配到的相邻 producer-consumer 中间张量访问；
3. 新延迟按融合链的 `max(sum(compute), adjusted memory, sum(communication))` 计算。

支持的实验：

- `Logit -> Attend` 在线 attention：GenZ 已把 score tensor 放在片上，因此只计算片上物化消除，不重复计算 HBM 节省；
- `QKV -> Logit -> Attend`：保留 QKV FLOPs，消除可以直接消费的 Q/K/V 中间 HBM 访问；
- `up+gate -> down`：假设 gate/up、激活和 down projection 按 token/tile 持久化，消除 MLP 中间 HBM 访问；
- 上述两类的组合场景。

结果同时包含聚合表和 `operator_instance_breakdown.csv`，后者逐层、逐去噪 iteration 列出算子延迟、前后相邻算子及 compute/memory utilization。

运行：

```bash
cd /home/zxf/vla-perf
python operator_fusion_prototype/analyze_operator_fusion.py \
  --profile-dir pipeline_prototype/results --gpu H100 \
  --output operator_fusion_prototype/results_h100
```

AdaRMSNorm、RoPE、Euler update 和 action input/output projection 在当前 GenZ π0 图中没有独立算子，因此报告只列为待建模候选，不虚构数值收益。
