# 几何监督数据 gate 结果

状态：`PASS_DATA_ALIGNMENT_ONLY`；进程退出码 0。无新模型训练、无 GPU 占用、无环境修改。

## 已执行

- 8 条 train 真实 LIBERO 轨迹、77 个唯一窗口全部导出；train/validation/test episode 集合无交叉。
- 验证轨迹 SHA256、原输入/目标 SHA256、窗口 RGB/动作/时间/物体与末端位姿/接触一致性。
- 验证正有限米制深度、旋转矩阵 SO(3)、末端相对位姿、相机刚体外参、内参形状/正焦距、接触矩阵对称且对角线为空。
- 深度原生坐标与 224×224 旋转版本明确分开，原内参不冒充变换后内参。未来几何只导出为监督 target。
- 未读取 validation/test 数据数组，不使用已看过的测试集选模。

## 接触覆盖（全训练轨迹帧，非 5 帧窗口统计）

| 轨迹 | 总帧数 | 窗口 | 夹爪接触帧 |
|---|---:|---:|---:|
| t0_e0_policy | 105 | 6 | 27 |
| t0_e0_hold_gripper_open | 221 | 13 | 162 |
| t0_e1_policy | 107 | 6 | 33 |
| t0_e1_hold_gripper_open | 221 | 13 | 170 |
| t1_e0_policy | 117 | 7 | 63 |
| t1_e0_hold_gripper_open | 221 | 13 | 89 |
| t1_e1_policy | 98 | 6 | 48 |
| t1_e1_hold_gripper_open | 221 | 13 | 161 |

张开夹爪失败轨迹也有接触，故 contact 不能当作 grasp/success；训练不能一律鼓励接触。
现有 `contact_min_distance` 非接触值为 0，不是完整 signed distance，不能据此生成穿透修复真值。负值可能包含接触求解器允许的重叠，需要碰撞几何与容差定义。

## 独立代码复核

experiment-bridge 要求的 gpt-5.6-sol/xhigh 只读审查发现清单完整性缺口；已修复并再次复核：8 train jobs、77 train rows、全局 ID 唯一、最终导出 77 个唯一 ID 且集合精确匹配。复核结论 BLOCKING=无。该复核是代码审查，不是实验物理效果审计。

## 证据与恢复

- 协议：`GEOMETRY_GATE_PROTOCOL_20260917.md`。
- 本地结果：`artifacts/gpu_gate_20260917/geometry_train_gate_result.json`，包含源/输出/脚本 hashes。
- 服务端输出：`__WAM2REPAIR_ROOT__/results/wam2repair/geometry_train_gate_20260917T094500Z/`。
- 日志：`__WAM2REPAIR_ROOT__/logs/wam2repair/geometry_train_gate_20260917T094500Z.log`。
- 命令：既有 `fastwam-py311/bin/python -u implementation/geometry_supervision_gate.py --root __WAM2REPAIR_ROOT__/results/wam2repair --out <新建个人结果目录>`；禁止 `python -O`，CPU 线程 2、超时 600 秒。

下一步尚未执行：投影坐标的 rendered-landmark 见证；RGB 主输出与深度/位姿辅助监督的统一模型最小单批训练；验证几何损失梯度影响 RGB 路径，而非仅改善独立几何 head。穿透 SDF/碰撞几何、物体对称旋转和输出 RGB 独立物理评估仍需实现。此 gate 不证明 WAM 物理错误或 VLA 收益，尚无深度修复效果结果。
