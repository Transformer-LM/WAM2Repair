# 本轮Goal逐项验收（未全部通过）

## 按原目标核验

| 原目标要求 | 当前证据 | 判定 |
|---|---|---|
| 个人目录/服务器/GPU规则 | 各启动日志即时GPU检查、个人PID身份、个人OSMesa环境；本轮GPU0/3任务已结束释放，未改系统EGL | 本轮执行有证据 |
| π0.5/LIBERO接口、权重来源 | official路径加载日志、各rollout同PID核验、下载MD5/CRC及16文件SHA256清单、真实动作/成功记录 | 已验证此次运行链；不代表benchmark性能 |
| WAM权重/VAE/时间接口 | 原错误加载记录保留；修复加载与VAE重建探针；完整151视频与manifest；16动作/5帧 | 已执行并留证 |
| 可辨认动作对齐WAM/仿真配对 | 首组及AI逐图审查场景可辨但有伪影；151窗口源hash、实际frames/actions/times/pose/contact逐项重验全部一致 | 配对工程验证通过；预测物理正确性未验证 |
| 最小统一视频修复单批学习 | video_overfit_result.json，loss0.41498→0.28983 | 已执行，仅学习能力 |
| episode隔离留出pilot | holdout_result.json/log，4/2/2组、77/38/36窗口，各方法1000steps；模型及输出hash | 已执行，单seed/小样本 |
| raw/copy/simple/no-WAM对照 | 五方法完整测试结果；repair输给copy/no-WAM | 已执行，必须保留负结果 |
| 正确未来保留检查 | 4低像素误差窗、oracle identity MAE0.0336072、盲审；低像素误差不是物理正确标签 | 压力检查已做，未证明保留 |
| 失败未来保留检查 | 26失败窗像素指标、AI盲审22未见明确成功虚构/4unknown；失败只来自hold-open | 部分；协议真人盲评未完成，不能验收语义保留 |
| 真实日志/失败/恢复状态 | 主报告、tracker、启动入口、审计原文/JSON、模型、数据hash、失败日志均保存 | 已保存 |
| 不冒充WAM/VLA主结果 | 独立审计WARN、claim partial；physical/WAM-specific/VLA/no，旧toy分开 | 报告明确遵守 |
| 前置问题先诊断，不跳gate | 修复随机backbone加载、base checkpoint能力、动作帧率；实际跑通后才训练 | 已留故障及修复证据 |

## 补证结果

`temporal_target_verified.json`：151窗口从原始轨迹重新生成全部target RGB并逐值相等，执行动作完全相等，实际时间及0.2秒采样间隔一致，contact/success/object/eef pose一致，无任何重复target hash或跨split重复。它只证明配对，不认证生成视频物理正确。

审计原文保留冻结时刻的unknown；上述补证和权重校验清单作为追加证据，不悄悄改写审计裁定。预训练WAM/normalizer的episode级重叠仍unknown，不能声称整个预训练链对这些任务未见。

## 仍需外部参与

协议指定的真人盲化复核尚无已填写记录。全部36匿名图及空表已准备；已向用户询问安排本人/同学复核，尚未收到答复。AI盲审明确不替代human_eval。不得自动填表、把unknown当通过，或以收窄目标的方式标Goal complete。

在此缺口处理前不启动新test调参、不把既有test继续用作新模型选择、不启动VLA闭环并宣称收益。若真人目前无法参与，需要用户决定如何保留/调整该未完成步骤；当前Goal仍未完整验收。
