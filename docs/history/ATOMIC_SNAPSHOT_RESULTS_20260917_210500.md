# LIBERO v2 同步快照前置结果

- Run：`atomic_snapshot_20260917T130232Z_2218503`，exit 0，19.8565 秒，CPU OSMesa，无 GPU 计算。
- 原始产物：`../artifacts/atomic_snapshot_20260917T130232Z_2218503/`。6 NPZ、2 XML、JSON 和完整运行日志已取回；远端原件保留。
- 程序状态：`PASS_ATOMIC_REPEATABILITY_ONLY`。本状态不表示整个 B0 通过。
- 2 场景 × 3 时刻 × 2 相机，每相机 RGB/depth/seg/K/外参及 body/geom/solver contact 两次读取逐数组 exact 相同；渲染前后 qpos/qvel/time 与 body/geom 不变。此处不是 controller 完整状态重置认证。
- 时刻 0.50/0.55/0.60 秒；physics dt 0.002 秒，实测控制间隔 0.05 秒。

## 任务盘点

| Suite | Task ID（默认 task order） | 名称 | 初始状态数 |
|---|---:|---|---:|
| libero_spatial | 0 | black bowl between plate and ramekin → plate | 50 |
| libero_object | 7 | milk → basket | 50 |
| libero_goal | 6 | cream cheese → bowl | 50 |
| libero_goal | 5 | push plate → front of stove | 50 |

本次仅前两任务 init 4 用于短 probe，非策略抓取轨迹。场景总 solver contact 数 87/29 包含所有接触，不是夹爪接触事件数量，不能由此宣称抓取覆盖。

## 未完成与后续

1. 独立 landmark 投影、跨视角深度/可见性认证。
2. visual/collision 几何映射、SDF 符号和 solver 容差标定。
3. 同身份点轨迹、接触子步事件、完整重置与动作分支检查。
4. 完整 D0 8 轨迹仍为 0/8；无新版 WAM 配对，无新训练、测试或 VLA 收益。

数据不会接入训练，直到相应 gate 完成。未来真实几何只作监督，不进入部署输入。独立代码复核已给 GO（same-family provisional）；结果完整性审计最终为 PASS with WARN，无完整性阻断，机器汇总 WARN。见 `ATOMIC_SNAPSHOT_AUDIT_20260917_211000.md`：原始文件/hash核验通过，但不能认证投影/物理语义/完整资产/跨运行确定性。
