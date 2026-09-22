# Research Output Manifest

> 本文件从本次恢复开始登记；既有历史见实验 tracker，不表示旧文件不存在。

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-09-17 17:45 | experiment-bridge | implementation/geometry_supervision_gate.py | implementation | CPU train-only 几何目标验证和导出 |
| 2026-09-17 17:45 | experiment-bridge | GEOMETRY_GATE_PROTOCOL_20260917.md | implementation | 数据 gate 协议与未完成能力边界 |
| 2026-09-17 17:45 | run-experiment | artifacts/gpu_gate_20260917/geometry_train_gate_result.json | implementation | 77 窗口实际运行结果和 hashes |
| 2026-09-17 17:45 | experiment-bridge | GEOMETRY_GATE_RESULTS_20260917_094500.md | implementation | 结果、独立代码审查和下一步 |
| 2026-09-17 17:45 | experiment-bridge | NEW_CHAT_START_PROMPT.md | implementation | 恢复入口更新 |
| 2026-09-17 17:45 | experiment-bridge | refine-logs/EXPERIMENT_TRACKER.md | implementation | 追加数据 gate 记录 |
| 2026-09-17 18:02 | experiment-bridge | implementation/depth_repair_sanity.py | implementation | RGB/深度同结构最小训练 |
| 2026-09-17 18:02 | run-experiment | implementation/run_depth_sanity.sh | implementation | UUID绑定、即时门禁单卡启动 |
| 2026-09-17 18:02 | experiment-bridge | DEPTH_SANITY_PROTOCOL_20260917.md | implementation | 3 train窗口两组300步协议 |
| 2026-09-17 18:02 | experiment-bridge | idea-stage/docs/research_contract.md | implementation | 当前研究契约，主张尚未证实 |
| 2026-09-17 18:02 | experiment-bridge | refine-logs/EXPERIMENT_PLAN.md | implementation | 最新深度诊断入口，旧计划保留 |
| 2026-09-17 18:02 | run-experiment | artifacts/gpu_gate_20260917/depth_sanity_result.json | implementation | 实际训练结果与配置hash |
| 2026-09-17 18:02 | run-experiment | artifacts/gpu_gate_20260917/depth_sanity_training.log | implementation | 完整训练曲线日志 |
| 2026-09-17 18:02 | experiment-bridge | DEPTH_SANITY_RESULTS_20260917_100228.md | implementation | 加深度暂未提升RGB的诊断结果 |
| 2026-09-17 18:02 | experiment-bridge | NEW_CHAT_START_PROMPT.md | implementation | 已完成训练恢复入口 |
| 2026-09-17 18:02 | experiment-bridge | refine-logs/EXPERIMENT_TRACKER.md | implementation | 追加两组实际训练结果 |
| 2026-09-17 18:02 | experiment-bridge | refine-logs/EXPERIMENT_CODE_REVIEW_20260917_100228.md | implementation | 部署阻塞修复及复核 |
| 2026-09-17 18:02 | experiment-bridge | refine-logs/EXPERIMENT_CODE_REVIEW.md | implementation | 最新代码复核别名 |
| 2026-09-17 19:58 | experiment-audit | DEPTH_SANITY_AUDIT_20260917.md | review | 上一轮最终WARN审计摘要 |
| 2026-09-17 19:58 | experiment-bridge | REGION_DEPTH_PROTOCOL_20260917.md | implementation | 三配置区域监督协议及失败修复 |
| 2026-09-17 19:58 | experiment-bridge | implementation/build_region_targets.py | implementation | CPU同周期分割标签重放 |
| 2026-09-17 19:58 | experiment-bridge | implementation/run_region_targets.sh | implementation | CPU OSMesa入口 |
| 2026-09-17 19:58 | experiment-bridge | implementation/run_region_depth.sh | implementation | 强制region目标的GPU入口 |
| 2026-09-17 19:58 | experiment-bridge | implementation/depth_repair_sanity.py | implementation | 增加区域损失与独立区域指标 |
| 2026-09-17 19:58 | experiment-bridge | implementation/run_depth_sanity.sh | implementation | 共用启动器region参数 |
| 2026-09-17 19:58 | run-experiment | artifacts/gpu_gate_20260917/region_targets_failed.log | implementation | 首次失败原日志保留 |
| 2026-09-17 19:58 | run-experiment | artifacts/gpu_gate_20260917/region_targets_result.json | implementation | 26张RGB exact与分割目标证据 |
| 2026-09-17 19:58 | run-experiment | artifacts/gpu_gate_20260917/region_depth_result.json | implementation | 三配置真实训练结果 |
| 2026-09-17 19:58 | run-experiment | artifacts/gpu_gate_20260917/region_depth_training.log | implementation | 完整loss曲线 |
| 2026-09-17 19:58 | experiment-bridge | REGION_DEPTH_RESULTS_20260917_115743.md | implementation | 局部depth改善未转为RGB优势 |
| 2026-09-17 19:58 | experiment-bridge | NEW_CHAT_START_PROMPT.md | implementation | 区域诊断恢复状态 |
| 2026-09-17 19:58 | experiment-bridge | refine-logs/EXPERIMENT_TRACKER.md | implementation | 追加区域训练记录 |
| 2026-09-17 19:58 | experiment-bridge | refine-logs/EXPERIMENT_PLAN.md | implementation | 区域对照执行入口 |
| 2026-09-17 19:58 | experiment-bridge | refine-logs/EXPERIMENT_CODE_REVIEW_20260917_115743.md | implementation | 区域代码复核与retry |
| 2026-09-17 19:58 | experiment-bridge | refine-logs/EXPERIMENT_CODE_REVIEW.md | implementation | 最新复核别名 |
| 2026-09-17 20:04 | experiment-audit | REGION_DEPTH_AUDIT_20260917.md | review | 六文件范围最终WARN |
| 2026-09-17 20:04 | experiment-audit | REGION_DEPTH_AUDIT_20260917.json | review | 审计状态/哈希/claim边界 |
| 2026-09-17 20:16 | experiment-plan | refine-logs/DATASET_DESIGN_20260917_201637.md | implementation | 4D数据v2设计，尚未采集 |
| 2026-09-17 20:16 | experiment-plan | refine-logs/DATASET_DESIGN_V2.md | implementation | v2设计固定别名 |
| 2026-09-17 20:16 | wam-research | refine-logs/METHOD_4D_REPAIR_20260917_201637.md | implementation | 统一时序几何到RGB网络候选 |
| 2026-09-17 20:16 | experiment-plan | refine-logs/EXPERIMENT_PLAN_20260917_201637.md | implementation | 数据优先五块计划，两项待证claim |
| 2026-09-17 20:16 | experiment-plan | refine-logs/EXPERIMENT_PLAN_PRE4D_20260917_201637.md | implementation | 旧执行入口归档 |
| 2026-09-17 20:16 | experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | 最新4D计划别名 |
| 2026-09-17 20:16 | experiment-plan | implementation/dataset_v2_contract.json | implementation | 数据字段/边界契约，实际v2轨迹0 |
| 2026-09-17 20:16 | experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | 追加R4D-00到04计划状态 |
| 2026-09-17 20:16 | experiment-plan | NEW_CHAT_START_PROMPT.md | implementation | 用户4D主线最高优先恢复入口 |
| 2026-09-17 21:05 | experiment-bridge | implementation/atomic_snapshot_probe.py | implementation | CPU原子快照前置脚本，非完整D0 |
| 2026-09-17 21:05 | run-experiment | implementation/run_atomic_snapshot_probe.sh | implementation | 个人OSMesa与CPU限时入口 |
| 2026-09-17 21:05 | experiment-bridge | refine-logs/ATOMIC_SNAPSHOT_PROTOCOL_20260917_205917.md | implementation | 两场景同步重复性协议 |
| 2026-09-17 21:05 | monitor-experiment | artifacts/atomic_snapshot_20260917T130232Z_2218503/ | implementation | 原始6NPZ、2XML、JSON、日志 |
| 2026-09-17 21:05 | monitor-experiment | refine-logs/ATOMIC_SNAPSHOT_RESULTS_20260917_210500.md | implementation | 程序重复性PASS，仅B0前置 |
| 2026-09-17 21:05 | experiment-bridge | refine-logs/EXPERIMENT_TRACKER_20260917_210500.md | implementation | v2前置执行状态快照 |
| 2026-09-17 21:05 | experiment-bridge | refine-logs/EXPERIMENT_TRACKER.md | implementation | 最新状态别名 |
| 2026-09-17 21:05 | experiment-bridge | NEW_CHAT_START_PROMPT.md | implementation | 最新CPU运行与未完成门禁 |
| 2026-09-17 21:10 | experiment-audit | refine-logs/ATOMIC_SNAPSHOT_AUDIT_20260917_211000.md | review | same-family provisional，无完整性阻断、范围WARN |
| 2026-09-17 21:10 | experiment-audit | refine-logs/ATOMIC_SNAPSHOT_AUDIT_20260917_211000.json | review | 审计机器摘要 |
| 2026-09-17 21:51 | experiment-plan | refine-logs/EXPERIMENT_PLAN_20260917_215105.md | implementation | 加入直接微调、同几何监督no-WAM、成本公平与WAM错误测量门禁 |
| 2026-09-17 21:51 | experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | 最新计划别名 |
| 2026-09-17 21:51 | experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260917_215105.md | implementation | 新对照状态及未部署代码记录 |
| 2026-09-17 21:51 | experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | 最新执行状态别名 |
| 2026-09-17 22:12 | experiment-bridge | implementation/geometry_ray_gate.py | implementation | 射线/深度CPU检查；经代码复核后运行 |
| 2026-09-17 22:12 | run-experiment | artifacts/geometry_ray_20260917T141018Z_2220873/ | implementation | FAIL_RAY_RASTER_GATE原始JSON与日志 |
| 2026-09-17 22:12 | monitor-experiment | refine-logs/GEOMETRY_RAY_RESULTS_20260917_221237.md | implementation | 3/12外部深度p95失败，未扩量 |
