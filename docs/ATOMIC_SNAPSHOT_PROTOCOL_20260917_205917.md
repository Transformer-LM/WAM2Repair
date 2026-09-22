# R4D-01a 同步快照前置检查

用户确认先做 LIBERO，不增加真实数据采集。本轮只执行 B0 的最小工程检查，不训练、不计完整 D0 轨迹。

- CPU OSMesa，复用个人环境，不修改 EGL，不调用 GPU policy。
- 盘点计划四任务的官方数字索引、初始状态数量与 BDDL hash。
- 碗和 milk 场景各取 init-state 4（development-only），seed 17，10 次零位移 settle；随后 3 个 tick，固定小幅 x 位移动作，仅用于验证同步读取。
- 每个 tick 显式 sim.forward 后，不推进仿真，顺序读取 RGB/depth/seg、相机矩阵、所有 body/geom pose 和 solver contact。
- 每个快照检查渲染前后 state/body/geom 完全一致；再次 forward/render 并逐数组检查 exact repeatability。
- 原生 256×256 图像只做 bottom-up→top-down；不复用旧 WAM 输出，不假设与旧 cached obs 像素一致。
- JSON 和每快照 NPZ、模型 XML 与 hash 全部保留；异常保存 FAIL 和 traceback。

允许结论仅为：测试场景在本路径下同状态渲染与几何读取可重复。不能认证投影误差、跨视角对应、collision/SDF 符号或接触覆盖；不能证明 repair/VLA 改善。完整 D0 仍为 0/8。

资源：2 CPU threads、0 张 GPU（CUDA_VISIBLE_DEVICES 为空，不是物理 GPU 编号 0），wall timeout 600 秒；无外网安装，无系统修改。代码须先经 experiment-bridge 要求的独立 reviewer 检查。
