# 模型转换使用说明

基于已完成精度验证的 `compiled.axmodel`，通过 SDK‑GEN 导出 AX650N 板端 C++ 可加载 `model.bin`。

plaintext







```
ONNX 输入      images [1,3,640,640] FP32 RGB / 255
AXModel输入    images [1,640,640,3] U8 NHWC RGB
共同输出       output0 [1, 8400, 24] FP32 (4*REG_MAX(16)+CLASS_NUM(5))
类别           good、broke、lose、uncovered、circle
```

## 1. 准备文件

仅使用已验证完成的 axmodel 作为输入，不再重新 build。

```bash
models/
  yolo11s-manhole-detection.axmodel
```

类别 ID：`0=good`、`1=broke`、`2=lose`、`3=uncovered`、`4=circle`

## 2. SDK‑GEN 导出 model.bin

当前 `pulsar2:4.0` docker 无 sdk‑gen，使用 Windows AX‑Model‑Tool：

1. 导入 `yolo11s-manhole-detection.axmodel`
2. 工具 → SDK‑GEN，运行时选择 `AX‑ENGINE Runtime binary`
3. 输出至 `deploy_out/`，得到 `model.bin`

```bash
docker run -it --net host --rm -v "${PWD}:/workflow" pulsar2:5.x
```

```bash
cd /workflow/model_convert
pulsar2 sdk‑gen \
  --axmodel ./model/yolo11s-manhole-detection.axmodel \
  --target_runtime axengine \
  --output ./deploy_out
```

## 3. 板端加载验证

拷贝 `model.bin` 至 AX650N 板端，C++ 接口初始化：

```cpp
IAIModel* model = CreateAIModel();
int ret = model->Init("./model.bin");
```

- `ret == 0`：加载成功
- `ret == -1`：加载失败

运行业务程序，效果应与 axmodel PC 仿真对齐。