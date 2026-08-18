# AX650N 模型转换工具

本目录恢复了可复现的 YOLO11 检测模型转换流程。当前示例模型接口：

```text
ONNX input : images  [1,3,640,640] FP32
ONNX output: output0 [1,9,8400] FP32
classes    : good, broke, lose, uncovered, circle
```

核心工具：

| 文件 | 作用 |
|---|---|
| `show.py` | 查看 ONNX I/O、节点和 Tensor，并运行 checker |
| `build.py` | 根据 ONNX 实际 I/O 生成 `build_config.json` |
| `cut.py` | 可选，按 Tensor 名裁切不兼容子图 |
| `pulsar2_sim/cli_detect_manhover.py` | AXModel 仿真的 RGB 预处理与检测后处理 |
| `SOP.md` | 从容器启动到质量验收的完整案例 |

模型和数据原件未随代码恢复。运行前放入：

```text
models/best_sim.onnx
dataset/calib_images/*.{jpg,jpeg,png}
```

然后从 `SOP.md` 第 1 步执行。
