# 上一轮深度最小训练完整性审计（最终返回摘要）

审计者 `/root/depth_integrity_audit` 与只读 fresh reviewer，gpt-5.6-sol ultra；same-family / provisional。最终 Overall WARN，替代先前 interim 状态。范围仅指定七个代码/协议/JSON/log文件，未访问服务器或重算原始NPZ。

| 检查 | 最终状态 | 证据与限制 |
|---|---|---|
| A GT provenance | WARN | depth_repair_sanity.py 当时61–94行：GT从bank/geometry加载，非WAM输出；原始trajectory未列入审计，不能独立重放上游 |
| B 自归一化 | PASS | 当时89–112行，RGB/temporal/log-depth直接均值与固定权重，无预测自身max/mean归一化 |
| C 文件/日志 | PASS（限定范围） | 全26条训练日志与JSON逐值一致；当时脚本及geometry hashes匹配 |
| D metric死代码 | PASS | 所报告metric均执行；temporal虽参与loss但上一轮未独立记录 |
| E 范围 | WARN | 同task0/episode0/source轨迹三个train窗口、单模型初始化seed17，两配置各300步；并非77窗口训练 |
| F 类型 | simulation_only + simulator real_gt | 非真实机器人、非held-out、非human evaluation |

允许结论仅为固定训练小批次上的接口/目标可以优化。两组RGB MAE分别0.0625736/0.0680309；加深度不优于RGB-only，二者也不优于copy0.0593994。深度log MAE下降、梯度非零不证明物理能力。不支持已收敛、泛化、多seed稳健、接触/穿透恢复、VLA收益或外部深度估计路线。

该审计对应 `depth_sanity_20260917T100228Z` 的代码版本；后续区域监督代码修改不在本审计覆盖内。后续新增 temporal 独立日志，保留原结果，不覆盖旧负结果。
