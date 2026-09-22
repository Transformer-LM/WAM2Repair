# 区域深度诊断完整性审计

最终结论：WARN。审计者 `/root/region_integrity`，gpt-5.6-sol ultra，fresh/same-family/provisional；仅六个指定文件，未访问服务器或修改文件。

| 检查 | 状态 | 证据 |
|---|---|---|
| A GT来源 | PASS（范围内） | build_region_targets.py:58，depth_repair_sanity.py:77：mask为同周期element sensor，RGB/state/time严格核对，深度来自独立target |
| B 自归一化 | PASS（范围内） | depth_repair_sanity.py:103、108：GT仅损失/评估使用，无预测min/max/std重标定 |
| C 文件与指标 | WARN | region_depth_result.json:18与log:7：代码/region JSON hashes匹配，39条日志与JSON逐值一致；原NPZ/PT未独立核验 |
| D metrics | WARN | depth_repair_sanity.py:138、170：loss/指标接线有效；COMPLETED与learnability flag不要求胜过raw/copy，不能读成效果PASS |
| E 范围 | WARN | depth_repair_sanity.py:53：同一episode前三train窗口、单seed300步，没有held-out或统计重复 |
| F 类型 | PASS | simulation_only，train-only learnability/overfit diagnostic；不是物理/控制验证 |

允许：采样时刻可见geom掩码与保存RGB/state/time对齐；固定3train窗口区域depth log MAE 1.34267→0.48663；代码/JSON/log内部一致与非零梯度连通。
不允许：加depth/ROI改善RGB修复、穿透/接触/控制改善、held-out泛化或显著性。RGB-only/full-depth/ROI-depth全图MAE为0.06257/0.06803/0.06757，均差于copy0.05940。不能称原NPZ与checkpoint已独立复验。

追踪：`.aris/traces/experiment-audit/2026-09-17_region/`，工作区根目录。
