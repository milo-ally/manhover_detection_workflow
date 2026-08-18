# 主码流

环境前提: 

- 主机，边缘设备处在同一个局域网
- 两边可以互相ping通

## 1. 主机安装 (MediaMTX + FFMpeg)

### 1.1 安装 FFmpeg

```bash 
sudo apt update 
sudo apt install ffmpeg -y 
```

校验是否安装成功

```bash
ffmpeg --version 
ffmpeg -encoders | grep libx264 # 出现 libx264 代表编码器正常
```

### 1.2 安装 MediaMTX

可以直接下载release二进制，省去编译

```bash
wget https://github.com/bluenviron/mediamtx/releases/download/v1.10.0/mediamtx_v1.10.0_linux_amd64.tar.gz
tar -xvzf mediamtx_v1.10.0_linux_amd64.tar.gz
```

解压包内直接包含 `mediamtx` 和 `mediamtx.yml`，不需要进入
`mediamtx_v1.10.0_linux_amd64` 子目录。MediaMTX 1.10.0 使用 `protocols`
配置 RTSP 传输方式；将配置文件中的这一行改为 TCP：

```yml
protocols: [tcp]
```

`paths` 保持默认的 `all_others:` 配置即可，不要改成文档旧版本中的
`paths: all:`。

### 1.3 主机防火墙放行8554端口

```bash
sudo ufw allow 8554/tcp
```

### 1.4 启动MediaMTX

```bash
./mediamtx mediamtx.yml
```

看到日志： `RTSP listener opened on :8554`，启动成功。这个终端窗口不要关闭。

### 1.5 主机推流 `test.mp4`

先获取主机在局域网中的 IP（把下面的 `HOST_IP` 替换成实际值）：

```bash
hostname -I
```

推流地址必须使用主机局域网 IP，不能使用只对本机有效的 `127.0.0.1`：

```bash
ffmpeg -re -stream_loop -1 -i ./test.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://HOST_IP:8554/src_in
```

如果当前目录不是 `mediamtx` 和 `test.mp4` 所在目录，请使用绝对路径，或先执行：

```bash
cd /path/to/model_deployment
```

### 1.6 主机本地验证输入流

```bash
ffplay rtsp://127.0.0.1:8554/src_in -rtsp_transport tcp
```

可以正常播放视频，代表上游链路没问题。无图形界面时可使用无窗口校验：

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8554/src_in
```

最后，边缘设备应读取：
`rtsp://HOST_IP:8554/src_in`。主机上的 MediaMTX 和 FFmpeg 推流终端都不要关闭。

## 2. 边缘设备

### 2.1 安装ffmpeg

```bash
sudo apt update 
sudo apt install ffmpeg -y 
```

### 2.2 设备网络校验

```bash 
ping HOST_IP
ffprobe -v error -rtsp_transport tcp -show_entries stream=index,codec_name,width,height -of default=noprint_wrappers=1 rtsp://192.168.0.129:8554/src_in 
# 预计输出: 
## index=0
## codec_name=hevc
## width=1138
## height=720
## index=1
## codec_name=aac
```

### 2.3 边缘设备 Python 最小 demo

保存下面脚本为 `rtsp_demo.py`：

```python
import cv2
import subprocess

HOST_IP = "192.168.0.129"
INPUT_URL = f"rtsp://{HOST_IP}:8554/src_in"
OUTPUT_URL = f"rtsp://{HOST_IP}:8554/ai_out"

cap = cv2.VideoCapture(INPUT_URL)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ffmpeg = subprocess.Popen([
    "ffmpeg", "-f", "rawvideo", "-pix_fmt", "bgr24",
    "-s", f"{width}x{height}", "-r", "25", "-i", "-",
    "-an", "-c:v", "h264_rkmpp", "-f", "rtsp",
    "-rtsp_transport", "tcp", OUTPUT_URL,
], stdin=subprocess.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取流失败")
        break

    # 在这里插入 YOLO 推理，并在 frame 上绘制检测框。
    ffmpeg.stdin.write(frame.tobytes())

cap.release()
ffmpeg.stdin.close()
ffmpeg.wait()
```

运行：

```bash
python3 rtsp_demo.py
```

主机上使用 `ffplay` 验证输出画面：

```bash
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/ai_out
```

无图形界面时使用 `ffprobe` 验证输出流：

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8554/ai_out
```
