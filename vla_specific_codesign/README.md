# Exact VLA Operator–Pipeline Co-Design

该目录是独立于原始VLA-Perf/GenZ和早期通用三轴实验的新原型。论文主线从
“流水线+融合+跨帧重用”收敛为严格保持当前帧计算等价的三项共设计：

1. `wavefront.py`：显式构建18层Prefix→Action依赖，调度
   `P0; (P1||A0); ...; A17`；
2. `kv_streaming.py`：Prefix QKV产生K/V tile后直接释放第一次Action
   Attention，同时保留KV cache写回和后续9次HBM读取；
3. `denoise_boundary.py`：融合Action OutProj、Euler Update和下一步Action
   InProj，保持FLOPs不变并减少中间物化和kernel launch。

普通QKV/online-attention和gated-MLP融合只作为supporting optimization，不声明
为核心创新。跨帧复用不进入严格等价主线。

## 运行

```bash
cd /home/zxf/vla-perf
python -m vla_specific_codesign.run_experiment \
  --profile-dir pipeline_prototype/results \
  --gpu H100 --bits bf16 \
  --overlap-efficiency 0.8 --kernel-launch-us 3 \
  --output vla_specific_codesign/results_h100

python -m vla_specific_codesign.run_experiment \
  --profile-dir pipeline_prototype/results_rtx_4090 \
  --gpu RTX_4090 --bits bf16 \
  --output vla_specific_codesign/results_rtx_4090

python -m unittest vla_specific_codesign.test_codesign -v
```

## 输出

- `REPORT_CN.md`：详细中文报告；
- `codesign_ablation.csv`：全部单项/组合消融；
- `wavefront_schedule.csv`：层级依赖安全调度；
- `wavefront_kv_schedule.csv`：QKV完成即释放K/V的细粒度调度；
- `streaming_kv_edges.csv`：逐层Prefix K/V流量；
- `denoise_boundary_details.csv`：融合边界的kernel/流量明细；
- `kernel_launch_sensitivity.csv`：0–10 us启动开销扫描；
- `denoise_step_sensitivity.csv`：4–20步去噪扫描；
- `overlap_efficiency_sensitivity.csv`：共运行效率扫描。

## 口径

这是扩展Roofline what-if实验，不是实际CUDA kernel计时。扩展基线显式加入
原GenZ图中缺少的action input/output projection和Euler update，再与融合方案
比较。`overlap_efficiency`和`kernel_launch_us`均为显式参数，最终论文数字必须
使用Nsight实测校准。
