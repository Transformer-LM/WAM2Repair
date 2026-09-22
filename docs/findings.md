# Research findings — unified video pilot

## 2026-09-17

独立result-to-claim裁定partial：pilot_executed=yes；physical repair、future preservation、WAM-specific benefit、VLA benefit均no。独立审计WARN，均same-family/provisional。

修复相对raw降低30.52%未来pixel MAE，但比copy/no-WAM差约7%；oracle-clean输入仍被改写。辅助changed-region指标与AI盲审存在探索性正信号，但不是物理修复认证，不能覆盖主基线负结果。

假设（未验证）：短时间窗口中的静态背景占比高；小型残差模型可能主要做平滑/朝当前帧收缩，而不是学到交互动力学。下一步不能仅扩大同一像素损失训练或用已查看test调参，应先完成真实接触/姿态关键帧的盲化评审，纳入no-WAM/copy物理一致性对照；新模型需新的未触碰test groups。

研究方向不变；不是改做toy/state-vector任务。当前人工复核待用户安排，不伪造人工结果，不由AI替代。详见VIDEO_PILOT_RESULTS_20260917.md和CLAIMS_FROM_VIDEO_RESULTS_20260917.md。
