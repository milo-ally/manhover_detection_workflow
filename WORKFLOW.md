# Data Preparing and Data Pre-processing 
## Source 
> manhover detection dataset: https://universe.roboflow.com/sideseeing/manhole-cover-dataset-yolo-62sri/dataset/1#

# Model Training 

# Model Exportation 

# Model Convertion 

## Source  
> convert tool: https://huggingface.co/AXERA-TECH/Pulsar2/resolve/main/4.0/ax_pulsar2_4.0.tar.gz 

## Load
```bash 
docker load-i ax_pulsar2_4.0.tar.gz
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