# 高分辨率实验进展

日期：2026-09-17。按用户授权实施，未启动新goal，未声称旧goal完成。

## 已执行

- 既有个人fastwam-py311环境，未安装包或修改环境。新增CLI分辨率和train-only sanity；原默认112×224保持兼容，旧checkpoint/结果未覆盖。
- 224×448、seed17、前三个train窗口、100步容量检查：初始loss0.0813722536，最终0.0711159930，下降12.60%，未达到预先20%门槛。峰值allocated显存1302481920 bytes。
- 为排查是否高分辨率特有，追加相同三窗口/seed/步数/目标的112×224诊断：初始0.0786730126，最终0.0709766969，下降9.78%，同样未通过；峰值344981504 bytes。
- 两次均正常完成前向/反向，输出JSON status=FAIL，非OOM/NaN。不同分辨率loss不可当作统一网格的质量排名；仅对各自起点观察训练下降。
- GPU3两次启动前均0MiB/0%/无compute进程，结束后释放。GPU2旧个人OpenPI PID2143996未动。

## 决策

完整1000步repair/no-WAM对照尚未启动。两种分辨率均未通过短预算检查，因此不能断言高分辨率退化、容量不足或研究方向失败。下一步应先在train-only诊断中明确更充分的收敛预算和学习曲线，再决定是否进入完整对照；保留本次失败，不事后改写门槛。未对test训练、选模或新评测。

## 运行与证据

- 高分辨率run：`__WAM2REPAIR_ROOT__/results/wam2repair/video_highres_sanity_20260917T085601Z`
- 低分辨率run：`__WAM2REPAIR_ROOT__/results/wam2repair/video_highres_sanity_112_20260917T090118Z`
- 本地：`artifacts/gpu_gate_20260917/highres_sanity_result.json`、`lowres_matched_sanity_result.json`
- 固定设计：`HIGHRES_PROTOCOL_20260917.md`
- 额外CPU工作已完成并拉回：`artifacts/gpu_gate_20260917/full_factual_review_20260917/`。共4条完整factual test轨迹，不是完整WAM预测，包含终止尾帧，原始每视角256×256；帧数221/76/221/108，时长11/3.75/11/5.35秒。GIF有调色板量化，仅作展示。原轨迹hash核对通过，CPU进程已退出。

## 查新对实验的影响

查新发现VLAW、World-VLA-Loop、WoW、ReDRAW、Feedback-WM、GEM-4D等相关工作。当前不临时加入新模型/损失。后续若形成方法主张，必须比较直接WM适配/微调与独立repair，匹配数据预算；证明保持动作条件、保留真实失败、正确未来不被损坏，并用新episode评价动作排序/闭环。参见SCOOP_CHECK_20260917.md。
