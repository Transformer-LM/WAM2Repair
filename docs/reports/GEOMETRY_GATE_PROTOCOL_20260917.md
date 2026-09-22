# 深度与几何监督数据 gate（2026-09-17）

本轮只执行新增几何训练前的数据 gate，不修改旧模型、旧划分或历史结论。
依据原 DATASET_DESIGN 的 factual RGB/PointMap/pose/contact 对齐要求及用户加入深度、旋转、位置、接触与穿透的授权。

CPU、既有个人 fastwam-py311 环境，最多 10 分钟；读取已完成 official_grid 的 train 轨迹，验证原始 SHA256、77 个训练窗口的 RGB/动作/时间/位姿/接触逐项对应。validation/test 不读取数据数组，只检查 manifest 的 episode 隔离。
检查有限正深度、SO(3)、相对位姿和接触矩阵；导出独立 target 文件。保留原生深度/K/相机外参；224 深度采用 180 度旋转后最近邻采样，不把原 K 当成变换后的 K。投影约定另待 landmark 渲染验证。

通过条件：全部训练窗口均通过数值与源文件验证并导出；任何失败停止，不绕过断言。仅标记 PASS_DATA_ALIGNMENT_ONLY，绝不是几何修复有效。

后续统一视频模型需要：RGB 主输出；深度辅助监督；位置与 SO(3) 旋转监督；真实接触状态监督；相对运动/速度时序监督；基于明确碰撞几何的穿透约束。几何监督必须影响 RGB 输出，不能只让独立 pose head 变好。未来真值不得输入推理。contact_min_distance 的非接触零值不能用作 SDF，接触亦不等于抓稳；穿透标签、对称物体旋转和输出 RGB 的独立几何评测尚未解决。无需修改 EGL。
