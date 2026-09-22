# WAM2Repair 统一视频修复 pilot：结果与证据边界

状态：真实采集、WAM生成、单批训练、episode隔离留出训练/评估已执行；独立审计WARN、result-to-claim为partial（pilot执行yes；物理/保留/WAM-specific/VLA主张no），均same-family/provisional。AI盲化视觉诊断已返回。人工盲化标注未执行。评估类型为 simulation_only，不是实体机器人结果。

## 数据与方法

官方 `pi05_libero` checkpoint，LIBERO spatial task0/1，episode0–3，seed7；每个episode执行policy及hold_gripper_open两种模式。16条真实轨迹中8条policy成功、8条干预失败；这是受控采集构成，不能当通用benchmark成功率。WAM和π0.5冻结。

151个完整窗口，16动作对应5帧(offset0/4/8/12/16)。train77、validation38、test36，独立episode组4/2/2。固定episode0/1训练、2验证、3测试，同初始episode两模式留在同split。所有源/目标hash、时间映射、重复input hash和split映射检查通过；没有跨split重复输入。短尾动作按协议排除，不能声称测试覆盖所有成功终止瞬间。

统一3D残差图像模型输入raw WAM视频、当前RGB/proprio和动作，输出完整修复视频。无WAM对照另行训练、同容量同预算。seed17，各1000steps，validation选模后才评估test。原始/复制当前帧/固定blend/repair/no-WAM均报告。最终模型与每个测试窗口预测NPZ已保存在远端个人results目录。

## 留出图像误差

112×224双视图拼接分辨率，未来帧MAE，先每episode平均再对两个test episode取平均。

| 方法 | episode宏平均MAE |
|---|---:|
| 原始WAM | 0.05159544 |
| 复制当前帧 | 0.03346069 |
| 0.25 WAM + 0.75 当前帧 | 0.03696186 |
| 统一修复 | 0.03584921 |
| 同容量无WAM模型 | 0.03344935 |

修复相对raw误差降低30.52%，但比复制当前帧高7.14%，比无WAM对照高7.17%。两个test episode方向一致；不支持WAM-specific修复优于简单基线。

配对MAE差(repair − baseline)的episode级精确重采样95%分位区间：对raw为[-0.01941042,-0.01208205]，对copy为[0.00169927,0.00307775]，对no-WAM为[0.00176219,0.00303753]。仅2个独立组，区间高度离散且不可靠，只作探索性描述，不作显著性、稳健性或跨任务泛化结论。未进行test调参或重训。

## 正确/失败未来保留检查

- 预定raw MAE≤0.03的切片有4窗口、2组；repair-minus-raw宏平均=-0.00808483，像素误差增加比例0；修改幅度MAE=0.01934835。低像素误差不等同物理正确，不能称“正确未来全部保留”。
- 26个窗口来自实际失败episode；repair-minus-raw宏平均=-0.01674500，像素误差增加比例0；修改幅度MAE=0.02978338。上述指标不能证明失败语义未被美化。
- 给模型输入真实干净未来的oracle identity压力测试：输出相对干净输入的宏平均MAE=0.03360721，而恒等映射应为0。模型会修改正确输入，不支持严格identity保留。该测试是特权输入压力诊断，不是可部署策略表现。
- 36个测试窗口独立AI盲化诊断已完成并在返回后解盲：repair更接近reference 20窗，raw更近6窗，持平6窗，无法排序4窗。26个失败窗中repair更近17、raw更近2、持平6、unknown1；修复的失败→成功语义有22窗未见明确证据、4窗unknown。unknown不能当保留成功，未见也不等于不存在。4个低raw误差窗为repair更近1、持平3。此审查未含copy/no-WAM，不支持相对它们的物理优势；它不是人工标注或几何物理认证。协议中的人工盲化复核仍未执行。

## 能与不能声称

能声称：真实模拟器配对上的统一视频修复小模型可以训练；该单seed、2个留出episode pilot中降低raw WAM像素误差，但输给copy/no-WAM基线。

不能声称：穿模/接触/姿态物理错误已被可靠修复；正确与失败未来均被语义保留；WAM特有信息带来收益；VLA动作排序或闭环成功率提高。WAM存在可辨场景上的纹理破坏与运动偏差，不能把所有图像差异都标成穿模。

## 主证据

- `artifacts/gpu_gate_20260917/official_grid_state.json`
- `artifacts/gpu_gate_20260917/policy_identity_logs/`
- `artifacts/gpu_gate_20260917/full_split_verified.json`
- `artifacts/gpu_gate_20260917/full_wam_result.json`
- `artifacts/gpu_gate_20260917/holdout_result.json`
- `artifacts/gpu_gate_20260917/holdout_training.log`
- `artifacts/gpu_gate_20260917/holdout_statistics.json`
- `artifacts/gpu_gate_20260917/blind_review/`

远端运行：`__WAM2REPAIR_ROOT__/results/wam2repair/video_holdout_20260917T050816Z`。本地统计首次尝试因默认Python缺numpy失败；未安装包，改用现有个人CPU环境完成并拉回结果。
