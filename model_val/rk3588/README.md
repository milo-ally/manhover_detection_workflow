# RK3588 模型验证

本目录比较同一份井盖验证集上的 ONNX 与 RKNN 结果。ONNX 使用 ONNX Runtime，
RKNN 使用 RK3588 板端 `rknn_toolkit_lite2`。两者共用 `validation_common.py`，
保持 letterbox、`output0 [1,9,8400]` 解码、分类别 NMS 和 mAP 计算一致。

类别固定为：`good,broke,lose,uncovered,circle`。

```text
ONNX: images [1,3,640,640] FP32 RGB / 255
RKNN: images [1,640,640,3] U8 RGB
```

验证数据存放在本目录 `images/` 和 `labels/`，配置见 `data_rknn.yaml`。
完整准备和运行命令见 `SOP.md`。
