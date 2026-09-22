# 可见机器人/物体区域深度监督诊断

承接 DEPTH_SANITY_PROTOCOL：仍为同一三个train窗口、同模型/seed17/300步/224×448。先用原始seed7与原始动作CPU重放，验证13个采样时刻qpos/qvel/time及两相机原生RGB逐像素一致；再导出MuJoCo可见geom分割，0背景、1机器人（含夹爪）、2 objects_dict非fixtures的对象。按body祖先映射几何体，遮挡区域不补造标签。任何重放不一致停止，不启用掩码。

对照：RGB-only；全图depth；区域depth。三者RGB/temporal损失一致，仅depth监督范围不同。区域depth取机器人与物体可见区域的log误差均值，lambda0.1。GT mask/depth只作监督/评估，不作为输入。报告全图与区域RGB/depth误差及raw/copy对照、temporal项和mask覆盖率。只训练诊断，不测试、不扩大数据、不声明物理修复有效。

CPU重放限600秒，既有OSMesa个人runtime；GPU训练限1800秒，检查物理GPU3并使用对应UUID绑定。不会修改系统或原始数据。本轮不启用三维投影/SDF/姿态/接触loss；分割只证明可见区域标签对齐，不能证明穿透或真实接触。

本轮GPU入口固定 `run_region_depth.sh <已PASS的region目录>`，强制恰好一个参数，再转交共用启动器。共用 `run_depth_sanity.sh` 无参数的旧双配置模式保留兼容，不是本轮入口。

首次CPU尝试 `region_targets_20260917T115410Z` 失败：重放obs RGB与源一致，但post-step额外render的RGB有133/196608通道元素差异。保留失败；修正为官方 `camera_segmentations='element'` 观测传感器，与RGB同一观测更新周期获取标签，不再混用缓存obs与post-step render。仍严格要求每个采样时刻的obs RGB逐像素相等、state/time数值一致；未降低误差门槛。使用官方element输出geom IDs，不再自行解码额外渲染的objtype/objid。
