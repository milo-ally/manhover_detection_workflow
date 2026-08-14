# Data Preparing and Data Pre-processing 
## Source 
> manhover detection dataset: https://universe.roboflow.com/liujunxiang/manhole-cover-zsmly/dataset/3

# Model Training 
## Environment
```bash
cd model_train

# dataset download
pip install roboflow

# pytorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# YOLO training
pip install ultralytics
```

## Workflow
Run scripts in order:

```bash
python 1-download_dataset.py
python 2-data-preprocessing.py
python 3-check_environment.py
python 4-check-dataset.py
python 5-train.py
python 6-evaluate.py
python 7-test.py
```

## Script Notes
- `1-download_dataset.py`: download Roboflow dataset version 3 in COCO format.
- `2-data-preprocessing.py`: convert COCO annotations to YOLO labels and generate `data.yaml`.
- `3-check_environment.py`: print Python, Ultralytics, PyTorch/CUDA, and GPU information.
- `4-check-dataset.py`: check image/label counts, class distribution, and obvious bbox issues.
- `5-train.py`: train YOLO with explicit default parameters.
- `6-evaluate.py`: evaluate `runs/manhole-cover-yolo11n/weights/best.pt` on the validation split.
- `7-test.py`: run prediction on the test images and save visual results.

## Outputs
Ignored by Git:

```text
model_train/Manhole-Cover-5Class-3/
model_train/runs/
model_train/*.pt
model_train/*.cache
```

Main trained model:

```text
model_train/runs/manhole-cover-yolo11n/weights/best.pt
```

# Model Exportation 

# Model Convertion 

## Source  
> convert tool: https://huggingface.co/AXERA-TECH/Pulsar2/resolve/main/4.0/ax_pulsar2_4.0.tar.gz 

## Load
```bash 
docker load -i ax_pulsar2_4.0.tar.gz
```

## Activate
```bash
# cmd: (将当前目录挂载到容器的/workflow目录下), 容器内执行的操作会影响宿主机
docker run -it --net host --rm -v "%cd%:/workflow" -w /workflow pulsar2:4.0

# PowerShell: (把当前目录挂在到/workflow目录下), 容器内执行的操作会影响宿主机
docker run -it --net host --rm -v "${PWD}:/workflow" -w /workflow pulsar2:4.0

# Linux/WSL: (把当前目录挂载到容器内/workflow目录下), 容器内执行的操作会影响宿主机
docker run -it --net host --rm -v "$(pwd):/workflow" -w /workflow pulsar2:4.0
```
