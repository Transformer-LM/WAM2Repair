# WAM 视频修复：CPU 数据实验记录

日期：2026-09-17。研究目标以用户最新说明为准：冻结 π0.5 + WAM，训练真正的未来图像/视频修复模型；显式状态作为辅助监督或评估，不替代视频输出。

## 已执行

1. 启动前检查四张 A100：约 31/31/39/31 GiB、99–100% 利用率，均不可用。所有新进程 `CUDA_VISIBLE_DEVICES=`，无 policy 请求，无 GPU 训练，无系统安装。
2. 用 CPU 扫描旧 E1 的全部匹配 WAM metadata/NPZ，生成逐样本对照图、SHA256 和 `inventory.json`。
3. 使用既有个人 OSMesa 环境重放 `prepared_goal03/task0` 的 8 个旧 StarVLA 动作，保存 9 帧和 qpos/qvel/body pose。执行前复查 GPU，CPU 线程限制为 2，超时上限 180 秒，exit 0。

## 数据盘点

- 11 个 WAM 输出版本，共 36 个端点配对，18 个源样本，7 个 suite/task/episode 分组；匹配文件和字段检查无失败。
- 不能把不同 WAM checkpoint/采样步数版本当作独立真实轨迹。
- 不同 seed 仍使用 episode 0；跨 seed 不自动构成独立初始布局。
- 已保存的 WAM NPZ 只有 primary/wrist 末帧；真实数据也只有当前帧和末帧，没有完整未来视频、原始 simulator state 或逐帧接触标签。
- 所有样本暂定 `diagnostic_only`，不制造 train/val/test 有效性。

## 视觉检查：关键负证据

本轮实际查看的五张双相机对照图：

- `wam_goal03_task0.png`
- `wam_spatial02_task0.png`
- `wam_goal05_s29_step5000_task4.png`
- `wam_goal05_s29_steps8_task4.png`
- `wam_goal03_s43_vae_task0.png`

五个样本的 WAM 图像均为彩色块状纹理，无法辨认机械臂和物体。可标记 `global_visual_generation_failure`，不能标记为已证实的局部穿模。其余 31 个端点本轮未逐一视觉判定，保留 UNREVIEWED。

36 个端点在主视角上的像素 MSE 均高于复制当前帧；这只是低层误差诊断，不是物理准确性或控制性能指标。

因此旧 15/18、12/18、13/18 的结果仍只支持劣质预测图像干预会改变策略行为；不能归因于穿模。图像混合可能通过恢复真实当前图像内容降低干扰，不能视为物理修复机制证据。

## CPU 重放

`prepared_goal03/task0`：初始 proprio 最大绝对误差 5.94e-8；两路相机的当前与末尾图像 MSE 均为 0；保存 9 帧。

这是单样本图像/动作重放一致性证据。由于旧数据未保存初始 qpos/qvel 和完整控制器状态，不能认证完整物理状态精确复原，也不推广到所有轨迹。状态标记 `REPLAYED_UNVERIFIED`。

## 静态接口检查

- 旧 WAM 配置 `num_frames=33, action_video_freq_ratio=4`；数据加载器按 `range(0,num_frames,ratio)` 采样图像。旧探针传入 8 步 action、请求 5 帧视频，时间对应关系不能默认成立，需继续核对模型内部 action grouping 和 padding。
- 旧数据执行前将 gripper 从策略值映射到 ±1，但 WAM 输入使用映射前动作；需与训练数据定义逐项核对，不能未经确认直接复用 π0.5 动作。
- 本地推理脚本的非 `state_dict` checkpoint 分支调用硬编码 `WAM_CKPT`，忽略 `--ckpt`，是潜在 provenance 问题；尚未检查远端历史运行代码/权重格式，不能断言影响了旧 step5000 结果。
- 当前 OpenPI launcher 使用 `pi05_libero` 配置 + `pi05_base_pytorch` 目录；目录 config.json 记录 horizon 50，而此前运行报告 horizon 10。基座身份、训练配置覆盖与 LIBERO 能力必须独立验证，当前仅能确认过去 infer 链路可运行。
- 本轮发现旧 WAM metadata 指向的 500/5000 权重文件目前仍存在。没有恢复任何已删除文件；文件存在不意味着权重有效或可生成合理未来。

## Gate 结论与后续

- CPU 数据读取/匹配：PASS。
- 单样本 CPU 图像重放：PASS（严格限定图像端点一致性）。
- 旧数据直接作为物理视频修复训练集：NOT ADMITTED；缺完整视频，已查看预测发生全局失真。
- π0.5 baseline、有效 WAM 生成、视频 repair 训练：PENDING。

下一 GPU gate 必须先验证有效 WAM 生成：真实图像 VAE encode/decode、权重加载记录、action/frame 时间映射、零/打乱/原始动作对照，并保存完整视频。区分生成管线故障与模型能力不足，目前不推定根因。通过后再采集 π0.5 分布下的接触片段；不要用彩色块状预测训练所谓穿模修复模型。

## 产物

- 本地：`artifacts/legacy_endpoint_audit_20260917/`，36 张对照图、inventory.json、replay_report.json。
- 远端：`__WAM2REPAIR_ROOT__/results/wam2repair/legacy_endpoint_audit_20260917/`。
- 重放：`__WAM2REPAIR_ROOT__/results/wam2repair/legacy_replay_cpu_20260917/`。
- 日志：`__WAM2REPAIR_ROOT__/logs/wam2repair/legacy_replay_cpu_20260917.log`。
- 脚本：`implementation/audit_legacy_image_pairs.py`、`replay_legacy_cpu.py`、`run_legacy_replay_cpu.sh`。
