# 原子快照完整性审计

审计：/root/atomic_integrity，GPT-5.6-Sol ultra，same-family / provisional。

总体：PASS with WARN（无完整性阻断）；机器汇总 WARN。仅支持继续 B0 工程验证，不支持物理修复结论。

- A PASS：源自 LIBERO simulator；原始 BDDL/init-state 未打包，来源 hash/初始状态身份尚未独立重算。
- B PASS：无评分归一化；RGB/seg 纵轴翻转，depth 官方转换为米；NPZ 类型及正有限值已核验。
- C PASS：6 NPZ、2 XML、代码 SHA256 全匹配；NPZ 可无 pickle 读取，17 数组，0.50/0.55/0.60 秒，body/geom维度38/270和26/119。
- D PASS：两次 snapshot 全数组 exact 断言实际接入，成功后才保存；第二份快照未保存，离线不能重比，仅由代码hash和执行日志支持。
- E WARN：两个场景单init单seed，符合工程范围；协议需明确米制转换，审计包未包含代码review trace，launcher未由结果hash绑定；外部mesh/texture未打包，资产未认证。
- F PASS：simulation_only 的进程内自一致性工程检查，不是模型性能实验；完整 D0=0。

允许：限定本次2场景3时刻的连续快照重复性、渲染期间所检查状态/几何未变、现存产物hash完整。
禁止：跨运行确定性、物理/语义正确性、投影/跨视角/collision/SDF/接触覆盖认证、完整D0、修复或VLA收益。

后续：补齐投影与collision gate；正式采集绑定来源/launcher/资产hash。代码review实际已完成，trace另存 `.aris/traces/experiment-bridge/2026-09-17_atomic/`，未纳入本次审计文件清单，不把此项记录存在性当作重审。

