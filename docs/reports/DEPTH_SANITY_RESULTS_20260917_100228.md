# 深度辅助 RGB 修复最小训练结果

运行 `depth_sanity_20260917T100228Z` 已完成，GPU3 已释放；GPU2旧服务未动。
训练期间约21.74秒，峰值 allocated 1,924,758,528 bytes。checkpoint与预测保存在个人远端结果目录；JSON与训练日志已拉回本地。

## 固定范围与实际指标

3 个 train 窗口（同一个episode的前三个窗口），每组300步，seed17，224×448，同结构与初始化。下表全部是训练集指标，不是留出效果，也不能与旧36测试窗口的数字直接比较。

| 方法 | 末步 RGB MAE | 末步深度 log MAE |
|---|---:|---:|
| 原始 WAM | 0.07630026 | 不适用 |
| 重复当前帧 | 0.05939943 | 不适用 |
| RGB-only 同结构对照 | 0.06257361 | 0.90245295（无深度监督） |
| RGB+depth | 0.06803094 | 0.71716857 |

两组初始化 RGB MAE=0.07630026，深度log MAE=0.90256941。带深度组末步 RGB 不如 RGB-only，也不如复制当前帧；不得写成加深度提升修复效果。训练曲线有波动，报告预定末步而不是事后挑最低值。

`learnability_check=true` 仅表示两组 RGB 比各自初始化降低，且有深度监督组的深度误差降低。深度loss到共享encoder梯度范数7.55494，RGB loss到depth head梯度范数0.00058245；证明计算图连通，不证明输出图像的几何正确。

## 已实现与未实现

已实现共享视频特征、正深度预测、预测深度条件化 RGB 残差输出、训练GT深度监督。未来深度不输入模型；未使用外部单目深度估计器。像素级旋转/resize对齐重新验证通过，本轮不使用K或三维投影。
尚未实现位置、SO(3)旋转、接触和SDF穿透损失，亦未验证identity保持或π0.5闭环。它是统一修复模型的第一步深度优化诊断，不是完整物理修复器。

## 证据

- `DEPTH_SANITY_PROTOCOL_20260917.md`
- `implementation/depth_repair_sanity.py`、`implementation/run_depth_sanity.sh`
- `artifacts/gpu_gate_20260917/depth_sanity_result.json`
- `artifacts/gpu_gate_20260917/depth_sanity_training.log`
- 远端：`__WAM2REPAIR_ROOT__/results/wam2repair/depth_sanity_20260917T100228Z/`
- 部署前审查已修复 GPU ordinal/UUID 映射问题；完整性审计另行记录，不将代码审查当作效果认证。

下一步：先补有效的几何/接触空间定位与输出端约束，再做训练集内优化诊断和identity压力检查；目前没有依据直接扩大到完整训练或声称深度方案优于RGB。

## 完整性审计状态

更新：最终行号版审计已返回，Overall WARN；摘要见 `DEPTH_SANITY_AUDIT_20260917.md`。下方“待返回”为历史记录。

只读独立子代理初步结论 WARN（same-family/provisional）：七个指定文件内部未发现伪目标、自归一化或日志/指标不一致，但只有同一轨迹的3个训练窗口、单seed；源采集器与官方数据身份未在此次限定文件范围中独立认证。temporal项未单独记录。初步回复已保存至工作区 `.aris/traces/experiment-audit/2026-09-17_depth_sanity/001-integrity.interim.md`；最终行号版待返回，不记为已完成最终审计。
