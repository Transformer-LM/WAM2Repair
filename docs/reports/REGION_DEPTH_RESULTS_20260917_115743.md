# 区域深度监督诊断：实际运行记录

## 运行与对齐

CPU `region_targets_20260917T115643Z` 已PASS：原seed7、原动作重放；13时刻×双相机共26个原生RGB逐像素一致，qpos/qvel/time通过1e-8检查。可见geom按body祖先分为机器人/非fixture对象/背景，采用与RGB同周期的官方element segmentation observable。
首次 `region_targets_20260917T115410Z` FAIL 已保留：额外post-step render与原obs不完全一致。未放宽RGB门槛，修复了观测取样方式。

GPU `depth_sanity_20260917T115743Z` 三组各300步已执行，退出码0，GPU3释放。三train窗口、同episode、seed17、224×448；同模型/初始化/优化器，仅depth loss范围不同。训练31.83秒，峰值allocated1,935,782,400 bytes。不读val/test。

## 末步原始指标（全部训练集，越低越好）

| 方法 | 全图RGB MAE | 机器人/物体区域RGB MAE | 区域depth log MAE |
|---|---:|---:|---:|
| 原始WAM | 0.07630026 | 0.12352411 | 不适用 |
| 当前帧复制 | 0.05939943 | 0.11074034 | 不适用 |
| RGB-only | 0.06257361 | 0.11034364 | 1.33158576（未监督） |
| 全图depth监督 | 0.06803094 | 0.11181087 | 1.14672124 |
| 区域depth监督 | 0.06757228 | 0.11681442 | 0.48662728 |

区域depth初始log MAE=1.34267008。区域深度更准确，但区域RGB仍比RGB-only与全图depth差，不得写成区域监督已改善物理修复。RGB-only区域MAE略低于copy，但只有同一轨迹3个训练窗口，不能作泛化/显著性/物理结论。

区域覆盖率（未来帧）整体28.59%，各窗口23.11%/26.87%/35.78%。GT masks只用于监督与区域评估，未输入网络；网络实际仍输出预测深度与RGB。区域是可见表面，不是接触/穿透标签。`learnability_check=true`与`roi_learnability_check=true`只说明损失可下降。

## 证据

- `artifacts/gpu_gate_20260917/region_targets_failed.log`
- `artifacts/gpu_gate_20260917/region_targets_result.json`
- `artifacts/gpu_gate_20260917/region_depth_result.json`
- `artifacts/gpu_gate_20260917/region_depth_training.log`
- 协议 `REGION_DEPTH_PROTOCOL_20260917.md`
- checkpoint与预测：`__WAM2REPAIR_ROOT__/results/wam2repair/depth_sanity_20260917T115743Z/`

部署前独立代码复核已通过，含强制region参数入口和第一次CPU失败后的修复复核。完整性审计进行中，不把代码审查当作效果审计。

更新：完整性审计已完成，最终WARN（same-family/provisional），见 `REGION_DEPTH_AUDIT_20260917.md/.json`。39条日志与结果一致；原NPZ/PT未纳入审计；只支持训练集优化诊断，不支持加深度改善RGB或物理/控制能力。

下一步应测RGB与深度目标在共享层的梯度尺度/方向，并检查RGB解码对几何的依赖；若有冲突，先做受控优化/结构干预，再扩充pose/contact/SDF。不可仅因depth指标下降就扩展为完整物理修复训练。当前没有接触、旋转、穿透或π0.5闭环新结果。
