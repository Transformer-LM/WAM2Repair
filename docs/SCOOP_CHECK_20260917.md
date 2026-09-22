# WAM2Repair 查新：生成未来修复与VLA

检索时间：2026-09-17。这是有边界的查新，不是不存在同类工作的证明。公开预印本/匿名投稿不自动等于已接收论文，本文不独立复现外部结果。

## 1. Verdict

**Level 2 — High Overlap（高重叠风险，非完全撞题）。** 在宽泛的“修正世界模型的物理预测错误以改善机器人控制”表述下，VLAW、World-VLA-Loop、WoW在问题、关键认识、机器人领域三轴相近，机制不同。ReDRAW使“冻结WM+轻量残差”也不能单独作为新颖性。当前没有确认某篇完整覆盖“冻结动作条件视频WAM、另训统一RGB视频repair、相同动作下保持真实失败、改善冻结π0.5候选排序”的全部组合；这只是尚未发现，不是唯一性认证。

## 2. Delta

不同于VLAW通过真实成功/失败rollout直接微调世界模型并继续训练π0.5，本方案拟冻结两者，在固定动作条件下单独学习未来视频修复，目标是在不把真实失败美化为成功的前提下减少交互预测错误并提高候选动作排序准确率；后两项收益尚待实验验证。

此delta是待验证目标，不是已实现贡献。若最终只有普通CNN像素残差和更低MAE，难以排除是常规后处理，也不足以支持论文主张。

## 3. Decomposed claim

- Problem framing：当前RGB/本体状态/候选动作/原始WAM视频→更忠实于相同动作真实后果的视频，评价接触、姿态、错误事件与控制用途。
- Core mechanism：冻结WAM和π0.5，训练独立统一视频repair；目前实现只是小型条件3D残差CNN，物理机制尚未确立。
- Key insight：应修模型错误而非动作本身的真实失败；同时检验正确未来保持与失败保真。
- Application domain：LIBERO视觉机器人操作，计划用π0.5候选排序/闭环，不是一般视频美化。

路线：R1/R2（解码视频/视频latent），计划U3（候选评价）；当前只完成像素pilot，尚未实现U3收益。非已完成的MBRL或VLA联合训练。

## 4. Structured papers（优先深读的七篇）

以下日期是首次预印本日期；仅对已读范围下结论。四轴粗粒度匹配用于风险排序，不是“相似百分比”。

### P1 VLAW

- Title：VLAW: Iterative Co-Improvement of Vision-Language-Action Policy and World Model
- Date：2026-02
- Source：[原文](https://arxiv.org/abs/2602.12063)；本地 `papers/wam2repair_vlaw.txt`
- Problem framing：世界模型的接触细节不准确、成功偏置妨碍VLA提升。
- Core mechanism：真实成功/失败rollout微调Ctrl-World，生成补充轨迹，奖励筛选并更新π0.5。
- Key insight：仅有成功示范不足以学习可靠的动作后果。
- Application domain：真实机器人接触丰富操作。
- Overlap score：3/4粗粒度轴；不是独立已生成RGB修复器。
- Closest-passage evidence：§4.1世界模型更新目标；§5.2事件级真实/预测结果对照；§6五类任务范围限制。
- Venue：本轮以arXiv预印本核对，不凭二手搜索确认会议接收。
- Assumptions & scope：需要在线真实rollout；WM与policy均更新；不能等同冻结VLA动作排序。

### P2 World-VLA-Loop

- Title：World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy
- Date：2026-02
- Source：[原文](https://arxiv.org/abs/2602.06508)；`papers/wam2repair_worldvlaloop.txt`
- Problem framing：动作执行细微错误时，生成未来仍幻觉成功，妨碍RL。
- Core mechanism：SANS成功/近成功失败数据，Cosmos视频与reward联合训练，OpenVLA-OFT GRPO与数据回流。
- Key insight：细粒度失败数据和动作—结果对齐很重要。
- Application domain：ManiSkill、LIBERO、真实操作。
- Overlap score：3/4；机制不同。
- Closest-passage evidence：§3.1 SANS；§3.2 state-aware simulator；§3.3 VLA RL。
- Venue：arXiv预印本。
- Assumptions & scope：改变世界模型与策略；不是保留固定动作、修复已生成视频的后处理。

### P3 WoW / SOPHIA

- Title：WoW: Towards a World omniscient World model Through Embodied Interaction
- Date：2025-09
- Source：[原文](https://arxiv.org/abs/2509.22642)；`papers/wam2repair_wow.txt`
- Problem framing：机器人生成视频出现碰撞/穿透/物体持续性等物理幻觉。
- Core mechanism：critic评估输出，refiner重写语言prompt再生成；逆动力学从视频恢复动作。
- Key insight：生成结果应接受物理检查与反馈修正。
- Application domain：机器人交互视频及动作生成。
- Overlap score：3/4；生成—检查—修正叙事高度接近。
- Closest-passage evidence：§4.2.2 prompt refiner明确提到防止穿过固体；§4.2.3 critic；§4.2.4重新生成循环。
- Venue：本轮以arXiv预印本核对。
- Assumptions & scope：可改变prompt并重新生成，未等价于固定低层动作的监督式RGB repair；长文仅重点检查§4.2及引言/范围。

### P4 ReDRAW

- Title：Adapting World Models with Latent-State Dynamics Residuals
- Date：2025-04
- Source：[原文](https://arxiv.org/abs/2504.02252)；[作者项目页](https://redraw-research.github.io/project/)；`papers/wam2repair_redraw.txt`
- Problem framing：源环境与目标环境动力学失配，需要低数据适配。
- Core mechanism：冻结DRAW，MLP残差修改latent转移分布logits，训练actor-critic。
- Key insight：稳定表征上学习小型动力学修正可避免全模型适配过拟合。
- Application domain：视觉DMC和真实Duckiebot循迹。
- Overlap score：2/4（宽泛残差机制、纠偏认识）；机器人窄领域与固定动作视频任务不同。
- Closest-passage evidence：§4.2式15–20；§5.1 DMC；§5.2 Duckiebot。
- Venue：本轮以arXiv预印本核对。
- Assumptions & scope：latent模型，非VLA或WAM生成RGB后处理；§4.2说明充分可观测及源表征覆盖假设。

### P5 Feedback World Model

- Title：Feedback World Model Enables Precise Guidance of Diffusion Policy
- Date：2026-05
- Source：[原文](https://arxiv.org/abs/2605.15705)；`papers/wam2repair_feedback.txt`
- Problem framing：部署分布变化导致预测偏差与错误策略引导。
- Core mechanism：维护feedback state，用已执行动作后的真实latent观测误差校正下一预测，action-aware energy引导Diffusion Policy。
- Key insight：预测纠偏应与动作可控部分及真实反馈相联系。
- Application domain：LIBERO-Plus、Robomimic、真实操作。
- Overlap score：3/4粗粒度问题/认识/领域；在线observer机制不同。
- Closest-passage evidence：§4.2式8–12；§4.3 action-aware guidance；实验限定任务与OOD设置。
- Venue：arXiv预印本。
- Assumptions & scope：使用过去已执行动作的真实反馈，不能在执行前获得候选动作的真实未来；预测输出为latent，非RGB统一修复器。

### P6 GEM-4D

- Title：GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation
- Date：2026-05（已查版本v4，2026-08）
- Source：[原文](https://arxiv.org/abs/2605.22882)；`papers/wam2repair_gem4d.txt`
- Problem framing：生成视频几何/跨帧对应错误，妨碍可靠动作提取。
- Core mechanism：几何基础模型特征蒸馏进入生成主干，训练时几何分支，推理去除分支；逆动力学恢复动作。
- Key insight：视觉真实不等于几何一致；深度、相机、运动需时序约束。
- Application domain：机器人视频预测与操作。
- Overlap score：3/4粗粒度问题/认识/领域；训练生成器而非独立后处理。
- Closest-passage evidence：§3.2 geometry alignment；§3.3 inverse dynamics；§4数据和深度/对应评价。
- Venue：arXiv预印本。
- Assumptions & scope：部分真实数据深度是估计值，模拟数据深度为GT；不能把所有几何评价称为物理真值。用户若采用深度辅助路线，须重点比较。

### P7 Off-Manifold Refinement

- Title：Off-Manifold Refinement: Guiding Video Generators with a Frozen World Model
- Date：2026-08
- Source：[原文](https://arxiv.org/abs/2608.29904)；`papers/wam2repair_offmanifold.txt`
- Problem framing：生成视频中的接触消失、违反物理的运动等。
- Core mechanism：冻结视频生成器/V-JEPA，训练latent→embedding adapter，在ODE采样中用surprise梯度引导latent。
- Key insight：借助外部预测先验修正生成轨迹，而非仅选漂亮样本。
- Application domain：一般物理视频生成，不是VLA闭环。
- Overlap score：2/4（物理纠偏问题与认识）；具体机制及应用不同。
- Closest-passage evidence：§3.2式4–8；§4 VideoPhy-2评测；§5明确预测可解释性不保证物理正确。
- Venue：arXiv页面注明BMVC2026 accepted；此处是作者页面状态，非独立会议名单复核。
- Assumptions & scope：采样中修正而非生成后RGB模型；无固定机器人动作保真或π0.5收益认证。

## 5. Comparison result

- Proposed work
  - Title：WAM2Repair（当前拟议主张）
  - Date：2026-09
  - Source：本项目
  - Problem framing：固定动作的未来视频错误修复。
  - Core mechanism：冻结WAM/π0.5，独立统一视频repair。
  - Key insight：真实失败必须保留，修预测而非美化任务结果。
  - Application domain：LIBERO、π0.5候选排序（待实现）。
- VLAW
  - Title / Date / Source：P1 / 2026-02 / https://arxiv.org/abs/2602.12063
  - Problem framing：物理预测错误妨碍VLA。
  - Core mechanism：直接WM与π0.5联合迭代适配，不同。
  - Key insight：成功/失败后果都应真实。
  - Application domain：机器人VLA。
  - 3轴匹配，Level2 High Overlap。
- World-VLA-Loop
  - Title / Date / Source：P2 / 2026-02 / https://arxiv.org/abs/2602.06508
  - Problem framing：错误动作的成功幻觉。
  - Core mechanism：SANS+视频reward联合训练+GRPO，不同。
  - Key insight：细微失败需要正确建模。
  - Application domain：机器人VLA。
  - 3轴匹配，Level2 High Overlap。
- WoW
  - Title / Date / Source：P3 / 2025-09 / https://arxiv.org/abs/2509.22642
  - Problem framing：视频物理幻觉。
  - Core mechanism：critic改prompt重新生成，不同。
  - Key insight：物理反馈纠正生成。
  - Application domain：机器人想象与控制。
  - 3轴匹配，Level2 High Overlap。
- ReDRAW
  - Title / Date / Source：P4 / 2025-04 / https://arxiv.org/abs/2504.02252
  - Problem framing：sim-to-real动力学适配，不同窄任务。
  - Core mechanism：冻结WM学残差，宽泛匹配但修latent logits不是RGB。
  - Key insight：低成本纠偏匹配。
  - Application domain：DMC/Duckiebot非VLA，窄领域不同。
  - 2轴匹配，Level3 Medium Overlap。
- Feedback-WM
  - Title / Date / Source：P5 / 2026-05 / https://arxiv.org/abs/2605.15705
  - Problem framing：错误预测误导控制。
  - Core mechanism：在线latent observer，不同。
  - Key insight：动作相关纠偏。
  - Application domain：机器人视觉操作。
  - 3轴匹配，Level2 High Overlap。
- GEM-4D
  - Title / Date / Source：P6 / 2026-05 / https://arxiv.org/abs/2605.22882
  - Problem framing：几何不一致未来妨碍动作。
  - Core mechanism：训练时几何蒸馏，不同。
  - Key insight：像素真实不足以支持控制。
  - Application domain：机器人视频操作。
  - 3轴匹配，Level2 High Overlap。
- OMR
  - Title / Date / Source：P7 / 2026-08 / https://arxiv.org/abs/2608.29904
  - Problem framing：视频物理错误。
  - Core mechanism：采样中surprise梯度，不同。
  - Key insight：修正生成轨迹的物理偏差。
  - Application domain：一般视频，不同。
  - 2轴匹配，Level3 Medium Overlap。

## 6. 其他必须注意的候选（未纳入七篇主深读）

- [VLA-in-the-Loop](https://openreview.net/pdf?id=aT4LG8c6DE)：搜索索引可读方法段，PDF下载403，两种URL均失败；不能声称已完整读取PDF。其corrector在抓取风险时生成成功视频再改动作，并非保持同一动作后果的预测修复。匿名ICLR2026投稿，接收状态未核实。
- [REVAMP作者页](https://revampcorl.github.io/REVAMP/)：动作/接触条件WM与Q头，可靠性触发真实采集再适配。页面自称CoRL2026投稿；未找到/读取完整论文，不以页面结果作独立事实认证。方向非常近。
- [RENEW](https://arxiv.org/abs/2607.14180)：已读PDF§2–3，是偏好监督转移动力学与主动查询；小型Jumanji/classic-control中用了synthetic oracle，不是VLA或真人A/B物理验证。不能把用户六个视觉偏好直接包装成这类训练。
- [Self-Correcting VLA](https://arxiv.org/abs/2602.21633)：摘要表明稀疏想象修动作，未确认RGB后修复。
- [RoboAlign-R1](https://arxiv.org/abs/2605.03821)：摘要表明奖励后训练与滑窗重编码改善机器人视频，不是已确认独立RGB repair。
- [Self-Refining Video Sampling](https://arxiv.org/abs/2601.18577)：采样自修正，一般视频；不能把视频物理修正首次提出。
- [τ0-WM](https://arxiv.org/abs/2606.01027)：联合视频动作与候选动作rectification；不能仅凭rectification一词认定修预测。
- [SelfEvoWM](https://openreview.net/pdf?id=lVn5vLOkjP)：索引摘要为生成—验证—修复、定向仿真补数更新WM；系统设计/早期失败报告，未深读。
- [Ontology-Grounded WM](https://arxiv.org/abs/2608.13901)：摘要为任务谓词诊断与verification-gated修正，并非已确认RGB视频物理修复。

## 7. 搜索覆盖与限制

paper-search三查询（2024–2026，每源每查询最多3）：
1. robot world model prediction physical error correction
2. vision language action world model
3. action conditioned video refinement physics

源返回：arxiv9，crossref9，DBLP/OpenAlex/OpenReview/SemanticScholar均0；18unique，0跨源合并。0不意味着无论文：SemanticScholar返回429，OpenAlex504，DBLP解析失败。另用实时web检索动作条件、post-generation、residual、repair、refinement，并回到原文验证。没有新增未核实model-recall论文。

原始错误类别：`429 Client Error`；`504 Server Error: Gateway Timeout`；`Expecting value: line 1 column 1 (char 0)`；OpenReview PDF `FAILED: HTTP Error 403: Forbidden`。本轮不是穷尽检索，不能用无命中证明新颖性。受影响源已通过web补检，但不能保证补齐全部。

### API全部18条（保持原排序，包括误检）

|#|Title|Year|Source/ID|词法score|初筛|
|---|---|---|---|---|---|
|1|World-Gymnast: Training Robots with Reinforcement Learning in a World Model|2026|arxiv:2602.02454|9|机器人WM内RL，非直接repair|
|2|GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation|2026|arxiv:2605.22882|7|进入深读|
|3|Compositional Context Fine-Tuning Vision-Language Model for Complex Assembly Action Understanding from Videos|2026|arxiv:2607.10797|7|标题初筛：视频理解，未全文核对|
|4|Inference-Time Attention Steering for Vision-Language-Action Driving Models|2026|arxiv:2608.17095|6|标题初筛：驾驶注意力，未全文核对|
|5|Vision-Language-Action Model for Electrical Power Operation Robots|未知|doi:10.22541/authorea.15004596/v1|6|标题初筛：非视频repair，摘要未复核|
|6|ChatVLA-2: Vision-Language-Action Model with Open-World Reasoning|2025|doi:10.52202/085713-1518|5|标题初筛：VLA推理，摘要未复核|
|7|Unified Video Action Model|2025|arxiv:2503.00200|5|摘要：联合video/action表征，不是预测后修复|
|8|Prediction-Correction Method for Nonlinear Error Model-Based Compensation of a Hybrid Machining Robot|2026|doi:10.1109/tmech.2026.3690103|5|标题初筛：加工机器人误差补偿|
|9|English Grammar Auto-Correction Robot based on Grammatical Error Generation Model|2024|doi:10.12694/scpe.v25i6.2171|4|明显语法纠错误检|
|10|Prompting Video-Language Foundation Models with Domain-specific Fine-grained Heuristics for Video Question Answering|2024|arxiv:2410.09380|4|标题初筛：视频问答|
|11|Self-Refining Video Sampling|2026|arxiv:2601.18577|4|生成采样纠偏，相关背景|
|12|Error correction of model analog forecasts using a linear inverse model for improving statistical ENSO prediction skill|未知|doi:10.5194/egusphere-egu26-16264|4|气候预测误检|
|13|Hierarchical Pre-Training of Vision Encoders with Large Language Model|2026|arxiv:2604.00086|3|标题初筛：表征预训练|
|14|MultiModal Action Conditioned Video Simulation|2025|doi:10.1109/iccv51701.2025.01315|3|动作条件视频背景，未深读|
|15|Multi-scale video-conditioned prompting for open-vocabulary action recognition|2026|doi:10.1016/j.patrec.2026.07.011|3|动作识别误检|
|16|Play Everywhere: A Temporal Logic based Game Environment Independent Approach for Playing Soccer with Robots|2024|arxiv:2405.12628|1|机器人足球，不是视频repair|
|17|Refinement and Polish: Through Iterative Guidance|2026|doi:10.1007/979-8-8688-3144-7_9|1|书章，未取得足够摘要，未知|
|18|[survey] Vision-Language-Action and Vision Language Models for Robot Manipulation: A Comprehensive Review Towards Real-World Applications|未知|doi:10.20944/preprints202606.0400.v1|7|综述，不用作方法原创性一手证据|

多数API噪声条目只有标题/元信息，本轮未取得其摘要，不把标题筛除伪装成全文排除。Citation字段大多返回0，不能解读为真实0引用，也不据此排作者影响力。

## 8. 对当前实验的具体影响

1. 不改方向，但不声称“首次WM物理修复用于VLA”。
2. 核心对照增加到未来方法阶段：同数据直接WM适配 vs 冻结WM+独立repair；本轮仅分辨率pilot，暂不额外启动大模型。
3. 输入同一动作，GT依该动作真实执行结果；失败不改成功。增加near-success失败，而非只有强制夹爪张开。
4. 同时检验正确未来identity、动作shuffle/counterfactual保真、接触事件/姿态与排序regret；不能只报PSNR/MAE或A/B审美。
5. 冻结π0.5的控制接口仍需明确实现。训练一个RGB修复器本身不会自动改变π0.5动作。
6. 后续决策：先做train-only充分收敛诊断；若无超越no-WAM/copy或物理事件信号，不扩大训练。论文差异以可测结果支撑，不能靠模块名称。
