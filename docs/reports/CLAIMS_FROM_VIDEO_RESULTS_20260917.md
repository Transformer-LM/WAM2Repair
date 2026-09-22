review_independence: `same-family`  
acceptance_status: `provisional`  
integrity_status: `unavailable`（独立实验审计与 AI 盲化诊断仍在进行，未视为通过）  
claim_supported: `partial`  
confidence: `high`（对当前证据边界）；pilot 执行完整性的置信度为 `medium`，待独立审计确认。

分项判定：

- `pilot_executed: yes`。这不是因代码存在而推定：`official_grid_state.json` 记录 16/16 轨迹完成、8 成功/8 失败及各自唯一哈希；`full_split_verified.json` 记录 77/38/36 train/validation/test 窗口、episode 隔离且无跨 split 重复；`holdout_result.json` 为 `EVALUATED`，含 repair/no-WAM 各 20 个 validation checkpoint 记录、36 个唯一测试样本/预测哈希；统计文件的 `source_sha256` 与当前 holdout result 完全一致，结果内两个代码哈希也与给定脚本一致。单批 overfit 产物亦有实际数值轨迹（loss 0.41498→0.28983），但只说明可训练。
- `physical/interaction repair claim: no`。
- `correct/failure future preservation claim: no`。
- `WAM-specific benefit claim: no`。
- `VLA benefit claim: no`。

实际支持：

- 在预定的 simulation-only、单 seed、2 个留出 episode pilot 中，repair 将 episode 宏平均未来像素 MAE 从 raw WAM 的 `0.051595` 降至 `0.035849`，相对下降 `30.52%`，两个 episode 方向一致。
- 预定 secondary changed-region MAE 有弱探索性信号：repair `0.145320`，优于 raw `0.161207`、copy `0.151389`、no-WAM `0.150913`；但只有两个独立组，且该像素掩码不代表物理错误。
- 实现可见 test 确在两种模型完成 validation 选模后才评估；no-WAM 为另行同容量、同预算训练。

不支持：

- 主 MAE 上 repair 输给 copy-current `0.033461` 和 no-WAM `0.033449`，分别差约 `7.14%`、`7.17%`；两个 episode 均同方向，因此没有 WAM 特有信息收益证据。
- oracle-clean identity MAE 为 `0.033607`，而严格保持应为 0；模型会显著改动干净未来，不能声称正确未来被保持。
- 26 个失败窗口虽都未增加像素 MAE，但未验证是否把失败语义“美化”为成功；4 个低 raw-error 窗口也不能充当物理正确样本。
- 没有穿模、接触、姿态或结果语义的人工真值评测；未证明模型真正利用 action conditioning。
- 没有动作排序、策略重规划或闭环成功率实验，故与 VLA 收益尚无直接证据。
- 单训练 seed、仅两个独立测试 episode；现有 bootstrap 范围只能描述，不能解释为显著性、稳健性或未见任务泛化。
- 数值预检只证明五个值确实存在于文件，不证明其语义主张。

建议收窄 claim：

> 在固定的 LIBERO spatial task0/1 episode-held-out 单 seed pilot 中，所配置的统一动作条件视频模型可训练，并相对 raw WAM 降低未来帧像素误差；但未优于 copy-current 或同容量 no-WAM，且 clean-oracle 测试显示非恒等修改。当前结果不构成物理/交互修复、正确或失败未来语义保持、WAM-specific 价值或 VLA 收益证据。

下一项最小有区分力的实验：

完成协议内、基于现有 36 个固定 test 窗口的独立人工盲化复核，不再训练模型：匿名比较 raw、repair、no-WAM、copy 与 factual GT，在 simulator 标记的接触/姿态关键帧上评定物理/交互一致性，并记录 26 个失败窗口的 failure→success 语义翻转及 4 个低误差窗口的误修。以 episode 为统计单位，预先要求 repair 在物理一致性上同时优于 raw 和 no-WAM，且不增加语义翻转/误修。它能最小成本区分“真实物理修复”与“像素平滑/当前帧复制”；即使通过，也仍需后续独立 VLA 排序或闭环实验才能支持 VLA 主张。
