# AX650N 模型转换 SOP

案例：将 `models/manhole-cover-yolo11s-production.onnx` 转换为 `output/manhole-cover-yolo11s-production/manhole-cover-yolo11s-production.axmodel`。除容器启动外，命令均在容器中执行。

## 1. 启动容器

宿主机 PowerShell：

```powershell
cd C:\Users\lys6076\Desktop\workspace\LYG_workflow_1\model_convert
docker load -i ax_pulsar2_4.0.tar.gz # 装载容器
docker run -it --net host --rm -v "$(pwd):/workflow" -w /workflow pulsar2:4.0 # 进入容器并且将当前目录挂载到容器的/workflow目录下
```

容器内：

```bash
cd /workflow
python --version
pulsar2 version
```

## 2. 生成量化数据包

量化图片必须是有代表性的业务图片，不使用验证集：

```bash
tar -cvf dataset/manhole_cover.tar -C dataset/calib_images .
tar -tf dataset/manhole_cover.tar | grep -Ei '\.(jpg|jpeg|png)$' | wc -l
```

## 3. 检查 ONNX

```bash
python show.py --onnx_model models/yolo11s-manhole-detection.onnx \
  --format json --output models/yolo11s-manhole-detection.onnx.json --check
```

预期为 `images [1,3,640,640]` 和 `output0 [1,9,8400]`。checker 通过只表示 ONNX 结构合法。

## 4. 生成配置

不手写 `build_config.json`：

```bash
python build.py --onnx_model models/yolo11s-manhole-detection.onnx \
  --output_config config/yolo11s-manhole-detection.onnx.build_config.json \
  --calibration_dataset ./dataset/manhole_cover.tar \
  --calibration_size 130 --npu_mode NPU1 --overwrite
```

## 5. 转换

```bash
mkdir -p output/yolo11s-manhole-detection
pulsar2 build --config config/yolo11s-manhole-detection.onnx.build_config.json \
  --input models/yolo11s-manhole-detection.onnx \
  --output_dir output/yolo11s-manhole-detection \
  --output_name yolo11s-manhole-detection.axmodel \
  --target_hardware AX650 --npu_mode NPU1 \
  --compiler.check 3 --compiler.check_mode CheckOutput \
  --compiler.check_cosine_simularity 0.999
```

成功且 `output0` 可用时不需要裁切。

## 6. 可选裁切

仅当 Pulsar2 不支持尾部算子、板端需要中间 Tensor 或需要定位问题时使用。先从 `yolo11s-manhole-detection.onnx.json` 找到 Tensor 名：

```bash
python cut.py --onnx_model models/yolo11s-manhole-detection.onnx \
  --output models/yolo11s-manhole-detection_head.onnx \
  --inputs images \
  --outputs /model.23/Concat_output_0 \
  --overwrite --list-on-error

python show.py --onnx_model models/yolo11s-manhole-detection.onnx --check
```

裁切后必须对新 ONNX 重新运行 `build.py` 和 `pulsar2 build`；被裁掉的后处理由板端 CPU 实现。

## 7. 仿真

```bash
# 预备
cd pulsar2_sim
mkdir -p models sim_inputs/1 sim_outputs
cp ../output/yolo11s-manhole-detection/yolo11s-manhole-detection.axmodel ./models
cp ../dataset/calib_images/well0_0031.jpg sim_images/1.jpg
echo 1 > list.txt

# 预处理 
python cli_detect_manhover.py --pre_processing \
  --image_path sim_images/1.jpg --axmodel_path models/yolo11s-manhole-detection.axmodel \
  --intermediate_path sim_inputs/1

# 推理
pulsar2 run --model models/yolo11s-manhole-detection.axmodel \
  --input_dir sim_inputs --output_dir sim_outputs --list list.txt

# 后处理
python cli_detect_manhover.py --post_processing \
  --image_path sim_images/1.jpg --axmodel_path models/yolo11s-manhole-detection.axmodel \
  --intermediate_path sim_outputs/1 --output_image 1_result.jpg
```

脚本输入为 `U8 NHWC RGB`，输出解释为 4 个 `cx/cy/w/h` 加 5 个类别分数。

## 8. 质量验收

必须满足：ONNX checker 通过、Pulsar2 成功、余弦相似度不低于 `0.999`、输出无 NaN/Inf。再使用 `../model_val` 的同一独立验证集分别计算 ONNX 与 AXModel 的 `mAP50`、`mAP50-95`，建议绝对下降不超过 `0.01`。最后在 AX650N 实板人工检查困难样本并连续推理至少 100 次。
