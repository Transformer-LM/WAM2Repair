# 新对话启动提示

## 2026-09-17 21:51 最新实验计划修订

用户质疑同LIBERO额外训练的独立价值，已更新 `refine-logs/EXPERIMENT_PLAN_20260917_215105.md` 及latest别名：加入相同几何监督no-WAM预测器、直接WAM轻量微调（几何监督可行版本）、同数据/同GPU预算双视角，先测真实WAM错误和独立评测coverage再训练。近期顺序仍B0标签→D0 8轨迹→WAM配对/解析→B1。`geometry_ray_gate.py`/`run_geometry_ray_gate.sh`本地已写但未审查/部署/运行；球体距离unit不代表LIBERO mesh通过。无新GPU任务。不要把设计或已编写代码写成实验通过。

## 2026-09-17 用户授权暂停 π0.5 服务

已确认账号 __WAM2REPAIR_USER__ 的 OpenPI PID 2143996、端口10098无已建立连接后，按用户要求发送 SIGTERM。复查进程不存在、10098不再监听、GPU2显存0 MiB；当时四卡显存/利用率均0且无compute进程。未删除权重/实验数据。下方“OpenPI服务存在”为历史记录；再次需要策略推理时须重新检查GPU并启动服务，不能假定端口仍可用。

## 2026-09-17 21:05 最新：用户确认先做LIBERO，CPU原子快照检查完成

审计补充：最终结果审计已返回 PASS with WARN，无完整性阻断；6NPZ/2XML/代码hash均独立核验。仅进程内重复性通过，非物理语义认证。详见 `refine-logs/ATOMIC_SNAPSHOT_AUDIT_20260917_211000.md`；下面“审计进行中”为当时记录。

`atomic_snapshot_20260917T130232Z_2218503` 已完成（19.8565秒、无GPU）：2场景×3快照×2相机，同状态重复读取exact，渲染前后qpos/qvel/time/body/geom不变。四候选task默认order索引spatial0/object7/goal6/goal5，各50初始状态。程序PASS_ATOMIC_REPEATABILITY_ONLY，不能当完整4D数据gate。详见 `refine-logs/ATOMIC_SNAPSHOT_RESULTS_20260917_210500.md`；原始NPZ/XML/JSON/log在artifacts同run目录。完整D0轨迹仍0/8，无新模型训练。下一步独立投影/跨视角/collision几何与容差检查，再采D0；不启动真实机器人/真实数据扩展。结果审计进行中。

## 2026-09-17 用户最新方向：先建真正4D数据集与统一网络（最高优先级）

用户强调深度用于4D几何、接触与穿透，要求先设计/制作数据集和搭建后续网络。最新 `refine-logs/EXPERIMENT_PLAN_20260917_201637.md`、`DATASET_DESIGN_20260917_201637.md`、`METHOD_4D_REPAIR_20260917_201637.md`。已写数据契约 `implementation/dataset_v2_contract.json`，状态设计未采集，v2实际轨迹0。计划先D0 8轨迹验证，再D1目标96轨迹（4任务×12初始状态×2模式，状态/任务ID与跨任务去重待锁定），旧test不作新确认。网络主线改为非特权depth/pose/track→统一时序几何修复→几何驱动warp/条件化RGB输出，非只depth辅助head。单目depth不能独自认证穿透，须有效collision几何/容差；接触与失败应保留。下一步是同步快照/投影/碰撞标签D0 gate，而非继续只调3窗口depth损失。旧cached obs与即时sim.data同刻性未认证，不能把旧数据直接称完整4D真值。未启动新GPU训练/全量采集。

## 2026-09-17 区域深度监督诊断已完成（优先最新）

`region_targets_20260917T115643Z` 已通过CPU原动作重放与26张RGB exact对齐；首次post-step render对齐FAIL已保留，改为官方segmentation observable同周期取样后通过。三组各300步 `depth_sanity_20260917T115743Z` 已完成且GPU3释放：全图RGB MAE RGB-only/global-depth/ROI-depth=0.06257361/0.06803094/0.06757228；区域RGB=0.11034364/0.11181087/0.11681442；区域depth=1.33159/1.14672/0.48663。区域深度学得更准，但RGB未优于RGB-only，不能宣称物理修复。仅同一episode三train窗口，无test。详情 `REGION_DEPTH_RESULTS_20260917_115743.md`。下一步测共享梯度尺度/方向与输出几何依赖，不直接扩大训练；pose/contact/SDF未实现。旧深度审计最终WARN已返回，见 `DEPTH_SANITY_AUDIT_20260917.md`。

## 2026-09-17 深度辅助最小训练已完成（优先最新）

`depth_sanity_20260917T100228Z`：GPU3实际完成RGB-only/RGB+depth同结构同初始化各300步，3个train窗口、seed17、224×448。RGB MAE：raw0.0763003/copy0.0593994/RGB-only0.0625736/RGB+depth0.0680309；加深度当前未优于RGB-only。深度log MAE从0.902569降至0.717169；梯度连通检查通过，但不是物理修复认证。GPU3已释放。详见 `DEPTH_SANITY_RESULTS_20260917_100228.md`；只训练数据诊断，无val/test，不与旧测试数字混比。已接入内部预测深度与RGB条件化，尚无外部深度估计器及位置/旋转/接触/SDF损失。不要再说深度模型尚未开训，也不要说完整物理能力已实现。

## 2026-09-17 深度与几何监督恢复（最新）

用户要求参考 GEM-4D/Gen2Real，统一修复 RGB 视频中的接触、旋转、位置、穿透及时间错误。最新已执行的是 CPU 几何监督数据 gate：8 train 轨迹、77 窗口全部通过对齐并导出 target；无新训练。见 `GEOMETRY_GATE_RESULTS_20260917_094500.md`。未来几何严禁作为推理输入；contact 不等于抓稳，contact_min_distance 不能直接当 SDF。下一步投影坐标见证和带几何监督的统一 RGB 修复最小训练，穿透几何/输出独立评测尚待实现。不要把数据 PASS 写成修复有效，也不要误称已加深度训练。旧 100 步 20% gate 是短预算诊断而非通用否定标准，历史失败保留。GPU 状态需每次重查。

## 2026-09-17 用户后续授权与新实验（优先于下方历史条目）

用户已要求继续高分辨率修复实验并查找高度相似工作。六个用户视觉偏好已记录且解盲均选repair，但不是物理标签，也不是全36窗口胜率。不要再要求用户必须看出低分辨率物理错误才能推进。

最新见 `HIGHRES_PROGRESS_20260917.md`、`HIGHRES_PROTOCOL_20260917.md`、`SCOOP_CHECK_20260917.md`：高/低分辨率相同三train窗口100步容量诊断均未达到20% loss下降门槛（12.60%/9.78%）。完整高分辨率训练未启动；不是OOM，也不能据此否定高分辨率。下一步先明确train-only充分收敛诊断预算，不使用已查看test调参。原自动pilot的独立审计与claim审查已完成，下面“等待独立审计”是历史状态。完整factual参考导出与预测修复证据分开。

## Active goal（2026-09-17，最新）

**当前Goal状态：blocked，非complete。** 自动pilot及补证均完成，但协议真人盲评连续未获安排/提交；空表已填0/36。等待用户提供人工记录或明确处理意见后恢复。不要重跑已经完成的训练，也不要把AI诊断替代人工标注。最新验收见GOAL_ACCEPTANCE_20260917.md。

用户已明确授权设置 goal 并开始 GPU 实验；读取 `UNIFIED_VIDEO_REPAIR_GOAL_20260917.md` 恢复当前运行。GPU 0/1/3 曾空闲，必须每次重查。
已修正旧 WAM 未加载视频主干预训练权重的配置错误；旧失真结果仍不得用作有效WAM物理错误证据。官方pi05_libero权重16文件SHA256已通过，真实LIBERO采集16条已完成；8正常策略成功、8张开夹爪干预失败，仅是受控pilot。
最新结果见 `VIDEO_PILOT_RESULTS_20260917.md`：151窗口，train77/val38/test36，episode组4/2/2无交叉。统一视频repair与同容量no-WAM留出训练均已执行，测试MAE repair0.0358492，raw0.0515954，copy0.0334607，no-WAM0.0334493。修复优于raw但输给copy/no-WAM；oracle-clean identity误差0.0336072，不能声称正确未来保持不变，更不能声称物理修复/VLA收益。
当前等待独立审计、AI盲化视觉诊断和result-to-claim审查；完整运行恢复见 `UNIFIED_VIDEO_REPAIR_GOAL_20260917.md` 顶部。人工盲化标注未执行；不要把AI检查写成人工或几何物理认证。下面旧CPU/GPU繁忙/尚未训练等段落为历史状态，不覆盖此最新条目。

## 用户最新目标与 CPU gate（2026-09-17，最高优先级）

用户明确要训练 **WAM 未来图像/视频修复模型**，修复穿模等错误；后续冻结 VLA 改用 **π0.5**。旧计划的显式状态修复不再作为最终输出目标。用户当前不安装 EGL，GPU 暂被他人使用，先运行 CPU 数据实验。

先读 `CPU_DATA_GATE_20260917.md`：已盘点 36 个旧 WAM 端点对（18 源样本、7 episode 分组），完成一个旧动作片段的 CPU 重放，当前和末尾双相机图像 MSE=0。但实际查看的五个 WAM 样本均为无法辨认场景的彩色纹理，且旧文件只保存末帧，不能作为穿模视频修复训练集。下一重要 gate 是有效 WAM 生成/时序/权重加载验证和 π0.5 LIBERO baseline；不能跳过这些检查训练“物理 repair”。详情和证据均在报告中。

## 2026-09-17 最新恢复进展（优先于下方旧状态）

个人 OSMesa CPU 渲染已通过独立复核；使用 `.aris/compute/direct-ssh-__REMOTE_HOST__.md` 的既有 runtime，无需系统修改。EGL 本身仍未修复。
已生成第一批真实 LIBERO factual alignment sanity：`__WAM2REPAIR_ROOT__/results/wam2repair/factual/factual_cpu_20260917T022452Z`，4 个脚本候选 × 8 步，`alignment_validation.json` 为 PASS。该批无抓取接触/成功样本，slip 未知，仅验证采集和对齐，不属于训练/测试 bank，不是 WAM/VLA 主结果。
最新检查四张 GPU 均忙；OpenPI PID 2143996、tmux `wam2repair_openpi`、10098 端口存在，未发起新推理。恢复时重新检查 GPU，不能沿用空闲假设。
下一 gate：接触丰富的 factual bank、VLA 候选、episode 分组划分及标签有效性，然后 factual 单批 overfit；Stage-2 仍未启动。详细记录见 `refine-logs/EXPERIMENT_TRACKER.md` 的 2026-09-17 条目。

请先读取以下文件，再继续实验：

1. `D:/VLA-WAM Research/ideaspark_run/policy-relevant-imagined-state-repair/WAM2REPAIR_HANDOFF.md`
2. `D:/VLA-WAM Research/ideaspark_run/policy-relevant-imagined-state-repair/CROSS_CONVERSATION_RECOVERY_20260916.md`
3. `D:/VLA-WAM Research/ideaspark_run/policy-relevant-imagined-state-repair/refine-logs/EXPERIMENT_PLAN_20260915_000000.md`
4. `D:/VLA-WAM Research/ideaspark_run/policy-relevant-imagined-state-repair/refine-logs/DATASET_DESIGN_20260915_000000.md`
5. `D:/VLA-WAM Research/ideaspark_run/policy-relevant-imagined-state-repair/refine-logs/EXPERIMENT_TRACKER.md`

当前研究：WAM2Repair。目标是训练两阶段 action-conditioned physical-state repair model，修复 WAM imagined future 中的接触错误、相对 6D 姿态错误、滑落和物理幻觉，并验证修复后的 imagined state 是否改善 VLA 候选动作排序和闭环执行。

目前已确认：

- E0 synthetic repair sanity 已通过；
- Stage-1 v3 synthetic corruption repair 已通过；
- pi0.5/OpenPI policy 已成功加载；
- OpenPI WebSocket 已成功完成两次离线 `infer()`，action shape 为 `(10, 7)`，第二次约 106 ms；
- LIBERO/MuJoCo EGL 渲染由于服务器没有可用 `EGL_EXT_platform_device` 失败；这是环境问题，不是模型结论；
- 目前尚未完成 factual state bank、WAM-specific Stage-2、候选动作排序和闭环 VLA 主实验；
- 不得把 E0/Stage-1 toy 结果写成 WAM/VLA 提升结论。

请从当前第一个未完成 gate 继续：先解决个人环境中的离屏渲染或选择已有数据集建立 factual state bank，然后做单批数据对齐 sanity，再进入 Stage-2。每一步先检查服务器 GPU 和进程，并保留失败日志。

## 服务器规则

- SSH 用户只能是 `__WAM2REPAIR_USER__`。
- Windows 客户端只能使用：`C:\Windows\System32\OpenSSH\ssh.exe`。
- 服务器：`__REMOTE_HOST__`。
- 私钥：`__SSH_IDENTITY_PATH__`。
- 示例：

```powershell
& 'C:\Windows\System32\OpenSSH\ssh.exe' `
  -i '__SSH_IDENTITY_PATH__' `
  -o BatchMode=yes `
  -o IdentitiesOnly=yes `
  __WAM2REPAIR_USER__@__REMOTE_HOST__
```

- 服务器只允许读写 `__WAM2REPAIR_ROOT__`，物理路径是 `__WAM2REPAIR_ROOT__`。
- 个人代码放 `__WAM2REPAIR_ROOT__/workspace`；日志、结果、缓存、checkpoint 也必须在 `__WAM2REPAIR_ROOT__` 下。
- 禁止 root、sudo、su、修改 `.bashrc`、全局 Conda、系统 CUDA/驱动、共享目录、其他用户目录和外网安装。
- 不启动真实机器人。
- 不使用 W&B。
- 已有个人环境：
  - `__WAM2REPAIR_ROOT__/conda/envs/openpi-py311`
  - `__WAM2REPAIR_ROOT__/conda/envs/libero-render`
  - `__WAM2REPAIR_ROOT__/venvs/libero-eval-py310`
- GPU 共 4 张 A100，物理编号 0–3。每次启动前立即运行：

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
```

- 只有显存 `<500 MiB`、利用率 `<=5%` 且没有 compute process 的 GPU 才能使用；默认优先 GPU 2、3。不要假设之前空闲的 GPU 仍然空闲。
- 目前 OpenPI server 使用 GPU 2；如需继续使用，先检查其 PID、端口和 tmux 会话。

## 当前 EGL 问题

服务器系统级 EGL/NVIDIA 驱动不能由普通用户修复，也不应尝试 sudo。当前 `/dev/dri/renderD128–131` 属于 `root:render`，用户不在 `render` 组；MuJoCo `MUJOCO_GL=egl` 因此无法初始化 device display。可在个人范围测试 `MUJOCO_GL=osmesa` 或已有虚拟显示后端；不要修改系统驱动、设备权限或共享配置。
