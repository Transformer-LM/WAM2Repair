# 今日初步小实验状态（2026-09-17）

## 已完成且可复现

- 新鲜、未使用旧 holdout 的 `libero_spatial/task0` episode 4、5、6 已由冻结官方 π0.5 通过个人端口 10117 实际执行并保存。
- 每条为 48 个真实执行动作；episode 4/5/6 均未成功。episode 4 记录 10 个 gripper-contact 帧，episode 6 记录 3 个。这些失败是事实目标，后续修复不得改写为成功。
- episode 4/5 的四个 16-action 窗口已封装为训练 bank；episode 6 的首个窗口已封装为未见测试 bank。训练/测试按 episode 隔离。
- GPU 2 上的 π0.5 服务仅用于采集，采集结束后已由启动者正常 `SIGTERM` 释放；GPU 3 上 WAM 进程也已停止。

## 未完成的关键阶段

冻结 WAM 的新窗口推理没有完成，因 FastWAM 加载时发现本地缺少 `Wan2.2_VAE.safetensors`，并试图从 ModelScope 下载。服务器规则禁止出站下载；发现后立刻终止了本进程，未将该下载产物作为实验依赖或结果使用。

因此今日尚不能诚实地声称“WAM future 被修复”，更不能声称 VLA 改善。当前完成的是可用于该实验的新鲜事实配对数据 gate；完整 WAM→repair→factual comparison 仍被**合规离线 WAM VAE 资产缺失**阻塞。

## 2026-09-18 勘误

上段“资产缺失”的诊断错误。原始离线 VAE 一直存在于 `__WAM2REPAIR_ROOT__/models/fastwam/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors`（mtime 2026-09-03，SHA-256 `0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996`）。此前可推理的启动器设置了 `DIFFSYNTH_SKIP_DOWNLOAD=true`、`DIFFSYNTH_MODEL_BASE_PATH=__WAM2REPAIR_ROOT__/models/fastwam` 和 `PYTHONPATH=.../FastWAM/src`；昨日的新命令遗漏了这些设置，因而错误地转向工作区默认 checkpoint 路径并触发下载。下载得到的工作区副本与原始文件 SHA-256 相同，但不作为实验所需资产。恢复实验应仅显式使用 9 月 3 日的原始个人模型资产和完整离线环境变量。

## 产物位置（服务器个人目录）

- trajectories: `__WAM2REPAIR_ROOT__/results/wam2repair/today_pi05_t0_e{4,5,6}_today_fresh_collect_20260917T143033Z_2222834`
- merged train bank: `__WAM2REPAIR_ROOT__/results/wam2repair/today_tiny_train_merged_20260917T143033Z_2222834`
- held-out test bank: `__WAM2REPAIR_ROOT__/results/wam2repair/today_bank_e6_today_fresh_collect_20260917T143033Z_2222834`

## 需要的恢复条件

由管理员或数据所有者将已核验来源和校验值的 VAE 放入 `__WAM2REPAIR_USER__` 个人目录的离线 FastWAM 预期路径，或提供一个已验证的完全离线本地路径；随后先以断网/离线模式做单窗口 WAM 生成 gate，再运行训练与 held-out repair。
