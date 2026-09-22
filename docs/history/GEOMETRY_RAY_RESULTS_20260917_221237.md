# B0 射线/深度一致性检查结果

Run：`geometry_ray_20260917T141018Z_2220873`，CPU/OSMesa，3.62秒，exit 2。产物：`artifacts/geometry_ray_20260917T141018Z_2220873/`；执行代码哈希 `78d008dad997162228429d02cf4111b461b12def28214f4de1fdfb1d5ac7801`。

## 结论：FAIL_RAY_RASTER_GATE

12个快照×相机记录中9个通过，3个失败。失败均为 `libero_object` 的 `agentview`，3个时刻重复：geom ID match=1.0，但depth p95=7.443mm，超过预先写入的5mm工程阈值。最大误差147.16mm出现在geom77；背景/墙面类像素也出现约9.396mm系统差。其余agentview spatial p95=2.498mm；两场景wrist p95=0.293–0.722mm。

错误垂直翻转负控均被拒绝，错误flip的geom match为0.210–0.710，说明检查对该坐标错误敏感。sphere analytic signed-distance unit已通过，但它只测试API，不是LIBERO collision mesh认证。

这不是“接触/穿透失败”或WAM错误结果。它证明目前保存深度与重建XML几何射线在一个场景的外部相机上尚不满足本轮预注册阈值。阈值不修改，不采D0、不训练。下一步：在未XML重建的原生环境重放同一state，分别比较保存K/外参与当前K/外参及重新render深度，定位是XML/camera约定、depth转换还是ray协议问题。
