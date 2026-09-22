# 统一视频修复 pilot 预定协议

本协议在官方π0.5数据采集和留出评估之前固定。先通过官方checkpoint最小真实任务基线，再扩大采集；不能用base失败episode替代。

## 数据

- LIBERO spatial task 0、1；每任务 episode 0、1、2、3，seed 7；共8个episode分组。
- 每组分别执行原始policy和hold_gripper_open动作干预。干预只修改实际执行动作，未来仍来自真实模拟器；不生成虚假成功/失败标签。两个模式共享split。
- train：两个任务的episode0、1；validation：episode2；test：episode3。分组键suite/task/episode。最终只可声称repair模型对留出episode的pilot表现，不是未见任务泛化。
- 每条轨迹固定每16步采样一个16-action窗口，GT视频为offset0/4/8/12/16；不按结果后验挑片段。短尾不足16步排除并记录原因。
- 保存实际动作、真实future RGB、当前proprio、simulator state/pose/contact、success trajectory。slip未定义则unknown，不凭接触或单帧图像捏造标签。
- 两种动作模式不保证分别成功/失败：标签以env.check_success为准。如果没有实际失败或没有接触覆盖，记录不足并补充预先说明的采集，不把期望标签当真值。

## 修复训练与对照

- WAM与π0.5冻结；WAM使用已修正的基础权重加载、16action/5video帧和真实执行动作。所有input/target单独保存及校验hash。
- 同一统一动作条件小型3D残差网络，不按错误类型拆模型；当前RGB/proprio、动作和raw WAM视频为唯一推理输入。
- train优化，validation选择checkpoint；test只做锁定配置评估，不以test调整训练。至少记录一个训练seed的工程pilot，正式稳健性结论仍需3 seeds。
- 首次holdout固定seed17、AdamW lr1e-3/weight_decay1e-4、1000 steps、batch4（小训练集时缩小）、梯度范数上限1；每50steps以validation未来像素MAE选最优，raw-WAM和no-WAM训练预算相同。目标为未来L1 + 0.1时间差分L1。输入/输出分辨率112x224（双视角横向拼接后缩小）。这是与单批gate分开的预定pilot配置。
- 必报raw WAM、copy current、anchored blend、repair，以及同容量不使用WAM输入的模型。no-WAM对照必须单独训练，不能仅在测试时清零替代。
- 主指标先定未来帧pixel MAE，辅以变化区域MAE、时间差分误差；仅说明图像预测误差，不等同于物理正确。
- 正确未来保留检查：oracle-clean future输入的identity压力测试，与原本低误差raw片段上的误修分开报告，前者是oracle诊断而非部署结果。
- 失败未来保留检查：按真实失败episode报告误差及人工盲化视觉检查；不能用网络生成的“看似成功”作为成功标签。模型若把失败轨迹改成抓取成功需单列记录。
- 按episode汇总，候选/窗口不是独立统计单位；小pilot置信区间和样本覆盖限制明确报告。

## 完成界限

预先固定保留检查的描述性切片：raw future pixel MAE ≤ 0.03 为低像素误差切片（不等同物理正确）；记录 repair-minus-raw MAE、修改幅度、像素误差增加比例，并按 episode 汇总。实际失败 episode 单列同类指标；空切片报告缺失，不事后放宽阈值。上述切片不参与选择 checkpoint。

完成pilot不等于repair优于基线。若copy-current/no-WAM一样好或更好，保留负结果，不声称WAM-specific修复必要性。物理错误纠正和VLA闭环收益仍需独立验证。
