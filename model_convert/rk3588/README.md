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

部署默认使用 FP RKNN。该模型的 `output0` 同时包含像素坐标和 `0~1` 类别分数，
整张输出做 INT8 量化时容易因坐标动态范围过大而把类别分数压成 0；因此当前
C/C++ 部署不要使用 INT8 产物。INT8 仍可用 `--dtype i8` 显式生成用于实验。

完整命令见 `SOP.md`。工具仓库放在 `third_party/`，已由 `.gitignore` 忽略；
仓库地址、提交号和 wheel 版本均写在 SOP 中，不依赖未记录的本地环境。
