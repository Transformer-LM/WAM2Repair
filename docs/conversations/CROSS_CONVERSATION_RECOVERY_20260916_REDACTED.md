# 跨对话恢复检查（2026-09-16）

本文件记录另一个对话卡死后，通过项目文件和远端个人目录恢复出的事实状态。

## 可复核事实

### 本地代码/计划

- WAM2Repair 计划位于 `refine-logs/EXPERIMENT_PLAN_20260915_000000.md`。
- 数据设计位于 `refine-logs/DATASET_DESIGN_20260915_000000.md`。
- Stage-1 v3 训练脚本位于 `implementation/stage1_repair_train_v3.py`。
- 本地最近新增了 OpenPI/FastWAM 接入文件：`launch_openpi_wam2repair.sh`、`openpi_direct_probe.py`、`openpi_policy.py`、`libero_policy_remote.py` 和 `fastwam_model_cfg.yaml`。

### 远端个人目录

- GPU 0、1、2、3 在检查时均为 0 MiB、0% 利用率，没有活动训练进程。
- `openpi_direct_probe.json` 显示 pi0.5 policy 创建成功：`action_dim=32`、`action_horizon=10`。
- `server.log` 显示 OpenPI server 曾成功监听 `0.0.0.0:10098`，但当前没有对应进程或监听端口。
- E0 retry 的真实日志为 `logs/wam2repair/e0_gpu2_retry1.log`，状态 PASS：pose RMSE `0.08210 -> 0.03285`，relative-pose RMSE `0.07118 -> 0.03103`，contact/slip accuracy 为 1.0/1.0。
- Stage-1 v3 的真实日志为 `logs/wam2repair/stage1_gpu3_v3.log`，状态 PASS：synthetic corruption repair only；pose RMSE `0.06025 -> 0.03954`，relative-pose RMSE `0.04967 -> 0.00380`，contact/slip accuracy 为 1.0/1.0。
- Stage-1 v1/v2 多个 retry 仍为 FAIL，不能删除，应保留为负结果。
- 当前没有发现远端 `results/wam2repair` 下可用于宣称主结果的 Stage-2、候选动作排序或闭环 VLA 结果。
- FastWAM 基础文件仍在个人 `models/fastwam`：Wan2.2 5B 分片、Wan2.1 tokenizer、ActionDiT 权重等；用户此前删除的旧个人 `checkpoints/` 不应恢复。

## 尚未完成

1. 真实/仿真 factual trajectory/state bank。
2. WAM-specific Stage-2 error bank 和修复训练。
3. raw WAM、oracle、repair、reject/replan 的候选动作 ranking 对比。
4. 冻结 pi0.5 的模拟闭环评估。
5. experiment-audit 和 result-to-claim。

## 恢复顺序

1. 重新检查远端 GPU 和进程。
2. 用已有个人环境做最小 OpenPI dummy inference，确认 server/client 输入输出链路；不启动真实机器人。
3. 生成最小 factual/synthetic 数据集并固定 train/val/test split。
4. 将 Stage-1 v3 的模型从 toy bank 升级到带时序和候选 action 的 state bank；先做单 batch overfit。
5. 只有在数据与 repair 指标通过后，才启动 WAM-specific Stage-2。

## 不能声称的结论

目前不能声称 WAM2Repair 已经改善了 WAM 预测、VLA 候选动作排序、闭环成功率或论文新颖性。已通过的两个 gate 只证明：代码和损失在合成小数据上可学习。
