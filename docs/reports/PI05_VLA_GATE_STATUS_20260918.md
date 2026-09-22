# π0.5 VLA-facing gate status

用户明确要求只使用 π0.5，且后续不运行复制当前帧条件。当前设计的三条件为真实观测、raw WAM future、repaired WAM future；不是 StarVLA，也不是 copy-frame。

最小单-context probe 已完成，详见 `PI05_IMAGINED_FRAME_PROBE_RESULTS_20260918.md`：factual/raw/repair 均真实成功（60/67/65 prefix后步数）。固定 diffusion-noise、runtime artifact provenance、相同图像分辨率路径、服务身份、失败记录和 UUID/OSMesa 启动器已完成。该结果是 post-hoc diagnosis，不是成功率结论。

结果—主张审查为 `claim_supported: no`（high confidence；same-family provisional；本 probe 无独立 integrity audit）。可支持的最强表述仅为：repair-only imagined frame 能在一个已查看 context 中改变冻结 π0.5 的首次动作，且真实执行成功；不支持 VLA 决策质量、成功率或物理修复因果结论。

下一 gate 是预先固定 D2 多 context×候选动作及非特权 scorer。单 context probe 只能报告 imagined-frame downstream sensitivity；不能报告通用 π0.5 成功率提升或候选排序收益。
