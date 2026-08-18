# RK3588 模型转换

本目录将五分类井盖检测 ONNX 转换为 RK3588 使用的 RKNN。板端部署仍在
`model_deployment/rk3588`，本阶段只负责转换和保存 `.rknn`。

模型约定：

```text
输入：images [1,3,640,640] FP32 RGB / 255
输出：output0 [1,9,8400] FP32，4 个 cx/cy/w/h + 5 类分数
类别：good,broke,lose,uncovered,circle
```

转换工具使用 Rockchip 官方 RKNN-Toolkit2 2.3.2，目标平台为 `rk3588`。
当前真实转换产物为：

```text
output/manhole-cover-yolo11s-production.rknn
```

这是 INT8 RKNN；Toolkit2 日志显示输入和 `output0` 默认类型均改为 INT8，后续
C/C++ 运行时必须按 tensor 属性处理量化参数，不能假设输出仍是 FP32。

完整命令见 `SOP.md`。工具仓库放在 `third_party/`，已由 `.gitignore` 忽略；
仓库地址、提交号和 wheel 版本均写在 SOP 中，不依赖未记录的本地环境。
