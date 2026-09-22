# 统一视频修复 Pilot 实验诚实度审计

**总体裁定：WARN**  
**审计独立性：same-family；acceptance_status：provisional。**

未发现伪造真值、自预测归一化、虚构结果或明显跨 split 泄漏。WARN 来自证据范围很小、旧状态文档过期、人工盲化未完成及部分完整性检查仍不充分。方法结果本身是负的：repair 优于 raw WAM，但输给 copy-current 和 no-WAM。

## A. 真值来源：PASS

- RGB、深度、pose、contact 均直接来自 LIBERO/MuJoCo observation 与 simulator state；slip 明确写成 unknown `-1`，没有由模型生成标签：`implementation/collect_physical_bank.py:44-75`。
- rollout success 来自 `env.check_success()`，实际动作经 `env.step` 执行：`implementation/pi05_libero_pilot.py:27-45`。
- target 视频由已执行轨迹构建，与 input 分文件保存：`implementation/build_aligned_video_bank.py:26-53`。
- WAM 生成器只打开 `input.npz`，并断言只有 current RGB/proprio/actions，不打开 target：`implementation/wam_video_diagnostic.py:25-45`。
- 服务端日志实际显示从 official `pi05_libero/params` 恢复并加载其 norm stats、监听 10099：`policy_identity_logs/pi05_libero_official_20260917T044030Z.log:14-21`；各 rollout 又核对同一 PID 的 official 启动命令，例如 `official_grid_t1_e3_policy_20260917.log:6-8`。
- 原客户端代码的 `--checkpoint` 本身只是报告字段，并不绑定服务端（`pi05_libero_pilot.py:14,33,54`）；但本次运行的服务端与每次 rollout 日志补足了实际身份链。冻结证据仍没有 official checkpoint 内容哈希，因此上游来源的密码学认证为 unknown，不影响“本次实际从该路径加载”的 provisional 判断。

## B. 归一化：PASS

- 图像缩放为固定 `/255` 或 `/127.5-1`，不是用预测自身最大值、均值或方差：`video_repair_overfit.py:27-29`、`wam_video_diagnostic.py:29-31`。
- action/proprio 使用外部固定数据统计：`wam_video_diagnostic.py:15-20`，不是当前预测统计。
- changed-region MAE 的分母来自 GT motion mask 像素数，而非预测幅值：`video_repair_holdout.py:79-88`。
- `raw_relative_to_copy` 是同时保留 raw/copy 绝对 MAE 后的描述性比值：`inspect_paired_videos.py:22-29`。
- 外部 normalization stats 的训练数据组成未在冻结证据中给出；是否与评估任务分布重叠为 unknown，但不存在自归一化造高分。

## C. 数字与结果存在性：WARN

数值链本身全部复算一致：

- 16/16 rollout 完成，8 policy 成功、8 hold-open 失败；日志中的 task、episode、seed、mode、checkpoint、steps、success、contact 与 `official_grid_state.json:3-196` 全部一致。
- split verifier 报告 77/38/36 windows、4/2/2 episode groups、无重复 input hash：`full_split_verified.json:2-26`。
- full WAM 为 `GENERATED_ALIGNED`、151 records：`full_wam_result.json:18-24,780`。
- holdout 的 bank/WAM/两份代码 hash 与当前冻结文件一致：`holdout_result.json:26-29`；训练日志两方法各有20次 validation、共1000 steps：`holdout_training.log:8-47`。
- contact diagnostic 的均值可由19行记录精确重算：`contact_pair_review.json:203-204`。
- 单批 overfit 数字和 PASS 判据一致，但 copy/anchored 均更好：`video_overfit_result.json:14-20`、`video_repair_overfit.py:63-73`。
- `holdout_statistics.json:2` 绑定当前 holdout result hash；报告中的 MAE、百分比、区间和切片值均与统计文件一致。

WARN 原因是旧状态文档已过时：`UNIFIED_VIDEO_REPAIR_GOAL_20260917.md:13` 和 `EXPERIMENT_TRACKER.md:17` 仍称 holdout 未运行，而 `holdout_result.json:244` 及 `VIDEO_PILOT_RESULTS_20260917.md:3` 已明确完成。应更新或把旧段落标成历史快照，避免当前状态冲突。

## D. 指标实际执行：PASS

- repair/no-WAM 均单独训练、validation 选模，并在两个 checkpoint 保存后才进入 test：`video_repair_holdout.py:44-70`。
- raw、copy、anchored、repair、no-WAM、temporal MAE、changed-region MAE、oracle identity、repair edit 与预定切片均实际调用并写入结果：`video_repair_holdout.py:70-124`、`holdout_result.json:244-1692`。
- 后处理统计实际覆盖 episode-level MAE、辅助指标、paired deltas、slice edit magnitude 与 exact cluster bootstrap：`summarize_holdout.py:12-51`、`holdout_statistics.json:3-501`。
- 人工盲化视觉复核尚未执行；36张匿名图只证明材料已准备。`VIDEO_PILOT_RESULTS_20260917.md:34` 和 `blind_review/README.txt:1` 对此标注正确。AI视觉诊断不能替代人工标注或物理认证。

## E. 场景、seed 与范围：WARN

实际范围为：

- LIBERO spatial task 0/1、episode 0–3、固定 rollout seed 7；8 episode groups、16 trajectories。
- repair 只有一个训练 seed 17。
- test 只有两个独立 episode groups；36 windows 是组内相关样本，不能作为36个独立统计单位。
- train/test 使用相同两个任务，仅 episode/init-state 留出；协议也只允许声称 held-out episode pilot，不能称未见任务泛化：`VIDEO_PILOT_PROTOCOL_20260917.md:7-9,18,24`。
- policy 成功与 hold-open 失败完全共线；26个失败 test windows 全来自 hold-open 干预。因此 failure slice 同时混杂 action mode、轨迹长度和失败状态，不能推出自然 policy failure 的语义保留。
- 两 episode 的 bootstrap 区间极度欠分辨；报告已正确拒绝显著性或稳健性解释：`VIDEO_PILOT_RESULTS_20260917.md:25-27`。

主要结果：

- episode-macro MAE：raw `0.05159544`、copy `0.03346069`、anchored `0.03696186`、repair `0.03584921`、no-WAM `0.03344935`：`holdout_result.json:1678-1683`。
- repair 比 raw 低30.52%，但比 no-WAM 高7.17%；两个 test episode 方向一致。36窗中 repair 36/36 优于 raw，却仅1/36优于 no-WAM。
- 辅助 changed-region MAE 有探索性正信号：repair `0.14532009`、no-WAM `0.15091281`，但它是辅助指标、仅两个组，不能覆盖主指标的负结论：`holdout_statistics.json:154-165,195-206`。
- oracle-clean identity macro MAE 为 `0.03360721`，恒等映射应为0，说明正确输入仍被明显修改：`holdout_statistics.json:316-327`。

## F. 评估分类：PASS

- 主数据与 holdout：**simulation_only，使用 factual simulator GT**；不是模型生成 target，也不是现实机器人 `real_gt`。
- pixel/temporal/changed-region MAE：对图像预测误差是直接指标，但对“物理错误已修复”只能算 **proxy**。
- success/contact：模拟器标签；contact 不等于 grasp，代码和报告均未混淆。
- 当前没有 human_eval、真实机器人 GT 或闭环 VLA 收益证据。报告的 `simulation_only` 分类正确：`holdout_statistics.json:495-501`、`VIDEO_PILOT_RESULTS_20260917.md:3,38-40`。

## 数据泄漏与边界：WARN

没有观察到已检查范围内的跨 split 泄漏：

- episode group 三分互斥：`video_repair_holdout.py:22-24`。
- test 仅在两模型完成 validation 选择并保存后执行：`video_repair_holdout.py:66-78`。
- verifier 检查预定 episode→split 映射、全部 input/target 文件 hash及跨 split重复 input：`verify_full_bank.py:13-26`。

仍有三项 unknown：

1. verifier 只对 `input_sha256` 做重复集合检查，没有对 target hash 做跨 split 唯一性检查：`verify_full_bank.py:21-24`。
2. verifier只核对声明的 offsets 与文件 hash，没有重新打开 target 验证实际 `times`/action-index 对齐；对齐主要依赖 builder 实现：`verify_full_bank.py:10,18-20`、`build_aligned_video_bank.py:28-48`。
3. WAM预训练数据与 combined normalization stats 的 episode级重叠未提供清单；因此不能声称整个 WAM 管线对数据本身完全未见。

## 必修问题

1. 保留负主结果：不得声称 WAM-specific 优势、最佳方法或 repair 优于简单基线。
2. 当前 test 已经被查看；任何据此改结构、阈值或超参的下一版模型都必须使用新的未触碰 test groups。
3. 完成人工盲化复核后，才能讨论成功/失败语义保留；AI审图不替代该 gate。
4. 更新 goal/tracker 的执行状态和负结果，消除“尚未训练”与已有结果的冲突。
5. 扩大到至少3训练 seeds、更多独立 episode/任务，并按 policy/hold-open 分层报告；当前不能给稳健性、跨任务或自然失败结论。
6. 发布级完整性应补：target重复检查、实际时间映射重验、WAM预训练/normalizer数据说明，以及在客户端结果中固化服务端 metadata/checkpoint hash。

## Claim 限制

可支持：

- 官方路径 checkpoint 驱动的 LIBERO simulator factual collection 已执行。
- 单seed、两个 held-out episode 的工程 pilot 中，repair 降低 raw WAM 像素误差并优于 anchored blend。
- changed-region 辅助指标存在值得后续验证的探索性信号。

不支持：

- repair 优于 copy-current/no-WAM；
- WAM 特有信息带来净收益；
- 物理接触、姿态、穿模或失败语义已修复；
- 正确 future 获得严格 identity 保留；
- unseen-task 泛化、统计稳健性、真实机器人效果、VLA排序或闭环成功率提升；
- 8/8 policy 成功作为 LIBERO benchmark 成功率。
