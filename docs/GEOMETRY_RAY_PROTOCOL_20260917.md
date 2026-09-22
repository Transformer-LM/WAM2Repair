# B0 射线/深度前置检查协议

数据：`atomic_snapshot_20260917T130232Z_2218503` 6个开发快照，两场景、双相机；仅原子重复性通过。

- 使用保存XML在相同个人环境MuJoCo3.2.3重建模型，检查所有引用资产可读并记录hash，恢复qpos/qvel/time。
- 几何姿态容差1e-5（XML序列化精度），不宣称控制器完整恢复。
- 在256×256图像上固定网格步长12、边界8像素，剔除3×3分割边缘；不按预测误差挑样本。
- 用保存K/外参计算pixel-center射线，原生mj_ray查询visual geometry；与保存的depth米制值及geom ID比较。
- 工程阈值：每视角至少100射线、全部命中、geom ID一致率>=95%、depth误差p95<=5mm；这些不是物理穿透判据，也不替代1pixel landmark gate。
- 错误垂直翻转是负对照；报告原始逐射线数据和全部失败，不通过调阈值抹除失败。
- sphere分离/接触/重叠的解析距离测试仅认证API符号，绝不代表LIBERO mesh或WAM物理结果。
- 限2CPU threads、600秒wall、0GPU；无环境安装/系统改动。

本检查后仍需要真实mesh距离/容差、跨视角与landmark、接触事件覆盖、完整D0。训练或VLA主结果尚无。
