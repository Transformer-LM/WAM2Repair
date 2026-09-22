# π0.5 imagined-frame probe（post-hoc 单 context）

运行结果目录：`__WAM2REPAIR_ROOT__/results/wam2repair/pi05_imagined_probe_20260918T052001Z_2251648`。

这是冻结 π0.5 的显式 observation-substitution 诊断，不是 π0.5 原生 WAM 输入，也不是候选排序或通用闭环成功率实验。episode6 已用于此前像素评估和检查，因此不属于 sealed confirmation。

| 首次 π0.5 图像查询 | 真实 LIBERO 成功 | prefix 后步数 |
|---|---:|---:|
| factual 当前观测 | 是 | 60 |
| raw WAM 未来帧 | 是 | 67 |
| repair-only WAM 未来帧 | 是 | 65 |

repair 比 raw 少 2 步，但仍比 factual 当前观测多 5 步。三者都成功，样本数为 1，不能声称修复提高 π0.5 成功率或证明物理错误修复帮助 VLA。

完整性检查：

- 三条件首次 noise SHA256 都是 `c0f4eb69…`; 后续按 query ordinal 共用同一 `(10,32)` 内部 π0.5 diffusion-noise bank；策略输出仍严格检查 `(10,7)`。
- factual prefix 的 qpos/qvel/sim_time 误差均为 0；两相机第16帧 RGB 均 exact、max_abs=0。
- runtime 输入 NPZ 仅含 `repair`，无 factual/raw 数组；完整输入/动作/noise artifact SHA256 在 `result.json`。
- raw/repaired 首段 action chunk 相对 factual 的 L2 分别为 3.70283 / 3.69521；这只是动作敏感性描述，不是任务增益。
- 服务只由本实验启动，结束后已 SIGTERM 并确认 GPU2 释放。此前两次失败均保留：路径逻辑/物理别名断言、以及错误使用外部 `(10,7)` 而非内部 `(10,32)` diffusion-noise。

下一步：先完成 D2 的预注册多 context×候选动作、非特权 scorer 和真实执行排序 gate；不能从本记录直接扩展为 VLA 主结论。
