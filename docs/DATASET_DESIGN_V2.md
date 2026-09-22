# WAM2Repair-4D 数据集 v2：先定义、后采集

状态：设计冻结候选，尚未生成 v2 数据。用户授权以深度/4D关系为主线，不继续把全图深度辅助loss作为完整方法。旧151窗口与所有负结果保留为legacy exploratory，不充当新的独立测试集。

## 1. 学习任务和原子样本

模型修复的是同一初始观测、同一实际执行动作下的 WAM 未来视频，不是把失败动作变成成功视频。

每条记录以 `(scene_group, context_state_hash, action_hash, wam_checkpoint_hash, generation_seed)` 唯一标识：

- `inputs/`：当前双相机RGB、当前可测机器人状态、语言、动作序列、真实WAM预测视频；可选从这些输入估计的depth/pointmap/pose/track与置信度。
- `targets/`：同动作实际执行得到的未来RGB、米制depth、可见几何分割、物体/robot-link变换、相机K与每帧外参、对应点、接触事件、碰撞几何标签、真实成功/失败。
- `assets/`：本场景物体visual/collision几何、robot link几何与关节定义、各对象对称性与所有hash。受控LIBERO第一版使用已知几何，应明确这是known-geometry假设，不能声称未知物体零样本。
- `meta/`：实际时间、控制与仿真子步时序、坐标/单位/图像变换、任务/初始状态/seed、代码版本、raw/WAM/目标hash、标签valid/confidence、分割归属与资产引用。

未来GT深度/位姿/掩码只作监督或明确oracle。当前可测robot joint positions可以扩展输入，但必须只取机器人关节，不能把包含物体状态的完整sim qpos输入模型；π0.5接口保持原样。

## 2. 4D真值：不是逐帧depth简单堆叠

`4D = 统一坐标中的3D几何 + 时间对应/运动`。

| 能力 | 必须采集的GT | 训练/评估注意事项 |
|---|---|---|
| 位置 | 对象与每个robot link的world pose、相对pose | 米制平移；不能仅用RGB MAE |
| 旋转 | SO(3)、对象对称群或等价姿态定义 | 非对称物体独立报告；圆碗绕对称轴不按可辨旋转错误计分 |
| 时间/运动 | 同一对象的canonical表面点ID、跨帧world位置、可见性/遮挡 | 用mesh局部点经pose得到真实对应，不用无身份点云最近邻冒充track |
| 接触 | 两侧geom/link/object ID、contact point/normal、solver distance，可获取时记录force；逐控制步+子步事件 | 瞬时接触不等于抓稳；零接触是有效标签，缺失是unknown |
| 穿透 | 全robot links–objects、object–object的collision几何；独立距离查询/有效SDF、容差 | 不能把旧contact_min_distance=0当间隙，也不能把所有负solver distance当错误 |
| 滑落/支撑 | 接触持续、相对运动、支撑对象与释放时刻 | 事件规则经检查后启用；未验证保持unknown |
| RGB可见一致性 | 同时刻RGB/depth/seg/K/extrinsics、跨视角遮挡 | 不能仅报告模型自己的pose/depth头变好 |

深度反投影需使用校准的K、图像变换与每帧相机外参。腕相机运动必须补偿；独立单目depth存在尺度/遮挡不确定性，不能把估计depth当GT，也不能仅凭可见表面证明有符号穿透。

## 3. 同步采集与时间粒度

- 新建v2采集器，不覆盖原始数据。每个采样时刻先固定sim状态，统一forward，随后在不推进仿真的同一快照中采RGB/depth/seg、相机/对象/robot transforms与状态；记录快照ID。执行新观测采样后需重新生成对应WAM输入，不能直接复用像素不同的旧预测。
- 原来snapshot混合cached obs与即时sim.data；此前只验证了同周期obs RGB/seg，**尚未证明所有旧RGB与geometry精确同刻**。作为待验证风险，不宣称旧标签全部错误。
- RGB/depth/poses至少每个控制tick保存（现有约20Hz，实际dt由日志验证）；接触事件若要描述tick之间发生/消失，需读取实际MuJoCo子步日志，不虚构时间精度。
- 第一版WAM仍用已验证16动作→5帧 `[0,4,8,12,16]` 接口，约0.8秒；dense factual用于事件标签，评估预测只在它实际输出的时刻进行。不能插值出密集帧后说WAM已预测接触瞬间。
- 保存整段episode，采样时覆盖接近、首次接触、搬运、释放、失败恢复。完整抓取上下文不等于完整WAM rollout。更长horizon须独立验证生成接口，不能把各自GT起点的窗口拼成连续预测。

## 4. 分层数据规模和划分

### D0：标签管线最小集

2类对象（现有碗+至少一个非对称包装物），每类2个新初始状态，每状态正常/受控扰动各一条：计划8条轨迹。全为development/train，不作测试。先验证4D标签、碰撞几何、投影/可见性与同初始状态动作分支的重置。

### D1：受控pilot目标

4任务×12新初始状态×2执行模式=96条计划轨迹，48个scene groups；不是96条独立初始状态。
候选任务（只确认本地BDDL存在，尚未验证策略在它们上的表现/初始状态数量）：

1. libero_spatial现有task0碗抓取/放置（接触基础）。
2. libero_object `pick_up_the_milk_and_place_it_in_the_basket`（非对称包装物位姿）。
3. libero_goal `put_the_cream_cheese_in_the_bowl`（相对姿态/容器关系）。
4. libero_goal `push_the_plate_to_the_front_of_the_stove`（持续接触/推移）。

任务索引、init-state ID及asset hash需采集前核验锁定，不假定文件名即数字task ID。D0可取D1中训练groups作为第一波，不额外增加测试暴露。
每任务按scene-group预分8 train / 2 val / 2 sealed test；得到64/16/16条轨迹。现有task0 episode0–3已参与旧实验，v2不用于sealed test；可用init-state必须先盘点。
同布局、同episode的所有动作分支、WAM随机种子、相邻窗口、合成派生都属于同split。跨任务若底层scene/init状态相同也合并group，不可仅用task名称隔离。若这一约束改变实际数量，以去重后的manifest为准，不伪补独立样本数。

两种执行模式是行为配置，不预先标success/failure。正常π0.5也可失败，受控扰动也可成功；一律使用真实结果。扰动按预先固定动作规则产生，并在模拟器执行，不能只修改视频。保留自然失败，不能只有强制张开夹爪一种失败。

### D2：动作因果/排序子集

另留至少8个新context，每个从完全相同快照执行4个固定候选action chunks；保存controller/internal state或使用已验证reset+prefix replay。只匹配qpos不足以自动证明控制器状态一致。候选来源/幅度先登记；每个候选都生成WAM并真实执行。此小集用于因果gate，不是主benchmark。其scene group不能跨D1 train/test。

## 5. 错误类别与采样

真实任务失败 != WAM预测错误。WAM把真实失败正确预测出来时，应保持该失败。
标签为multi-label：contact_miss、false_contact、translation_error、rotation_error、penetration、temporal_drift、object_identity_error；每项附valid/confidence，不能由“这是扰动失败轨迹”直接赋值。
GT有真实pose/contact不代表WAM视频中的这些量已经可读。预测端几何解析器需在真实holdout图像校准；低置信/严重遮挡的错误类型标unknown，报告coverage，不能删掉难例制造高分。

保存全部完整窗口索引。训练loader可增加事件窗口抽样权重并保存概率，不能复制它们伪增样本量。val/test按固定连续规则评价，并另报事件分层，不人为平衡测试错误类别。
主训练与评价来自真实WAM生成；合成几何扰动只可作为带来源标记的预训练/单元测试。干净GT视频作为identity训练对照时明确记录oracle augmentation，所有变换仅由train产生。

## 6. 数据gate

G0 schemas、hash、split、timestamps、有限数与有效mask。
G1 同快照RGB/depth/seg/pose/K/外参：三维landmark投影、跨视角depth一致性及遮挡检查；GT replay误差可重复。
G2 collision资产：可知正间隙/接触/明显重叠的几何单元测试，mesh单位/坐标/内外符号有效，solver容差由GT轨迹统计而不是测试预测调阈值。无效SDF不启用穿透loss。
G3 真实WAM样本对齐：同action hash、正确权重、时间接口；预测端错误标签只在置信度足够时有效。
G4 非对称旋转与接触正负事件具备实际覆盖；没有覆盖的能力不得写已训练。

只有D0这些gate通过才收集全部D1。当前新增的是设计，不是96条已完成的数据集。
