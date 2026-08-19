# RK3588 基础推流 FLOW1-SSH-TUNNEL

本文档是 RK3588 的基础 RTSP 输入/输出链路，使用 SSH 双向端口转发，
不依赖主机与板端互相 ping 通。RK3588 侧的井盖 AI 程序请使用同目录
`manhole_cover_detection` 和 `FLOW2.md`。

## 1. 主机准备 MediaMTX 和输入流

主机为 Windows，使用 Windows amd64 版 MediaMTX 和 FFmpeg。

### 1.1 MediaMTX 配置

将主机 `mediamtx.yml` 的 RTSP 配置设为：

```yml
protocols: [tcp]
rtspAddress: :554
```

`paths` 保持默认的 `all_others:` 配置。Windows 防火墙放行 `554/TCP`。

启动主机 MediaMTX，并保持窗口运行：

```powershell
.\mediamtx.exe mediamtx.yml
```

看到 `RTSP listener opened on :554` 即启动成功。

### 1.2 主机推送输入测试视频

主机本地推流地址必须是 `127.0.0.1:554`，不能使用主机局域网 IP：

```powershell
ffmpeg -re -stream_loop -1 -i .\test.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:554/src_in
```

主机本地验证输入：

```powershell
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:554/src_in
```

### 1.3 建立 SSH 双向隧道

新开独立 PowerShell 窗口，保持该窗口运行：

```powershell
ssh -R 8556:127.0.0.1:554 -L 8557:127.0.0.1:8554 root@172.19.30.3
```

- `-R 8556:127.0.0.1:554`：将主机输入端口映射到 RK3588 的 `8556`。
- `-L 8557:127.0.0.1:8554`：将 RK3588 输出端口映射到主机的 `8557`。

登录成功且没有 `remote port forwarding failed` 后，隧道才可用。
隧道窗口关闭后两个端口映射立即失效。

## 2. RK3588 板端准备

### 2.1 安装 FFmpeg

```bash
sudo apt update
sudo apt install -y ffmpeg
```

确认板端存在 Rockchip 硬件编码器：

```bash
ffmpeg -encoders | grep -E 'h264_rkmpp|hevc_rkmpp'
```

本工程在线输出使用 `h264_rkmpp`。不要用只包含软件编码器的 FFmpeg 替换
板端 multimedia FFmpeg。

### 2.2 部署板端 MediaMTX

板端 MediaMTX 只负责输出 AI 结果流，监听 `8554`：

```bash
wget https://github.com/bluenviron/mediamtx/releases/download/v1.10.0/mediamtx_v1.10.0_linux_arm64.tar.gz
tar -xvzf mediamtx_v1.10.0_linux_arm64.tar.gz
```

板端 `mediamtx.yml` 至少配置：

```yml
protocols: [tcp]
rtspAddress: :8554
```

启动并保持运行：

```bash
./mediamtx mediamtx.yml
```

### 2.3 验证 SSH 隧道输入

板端只能使用隧道本地端口 `8556`，不使用主机 IP，也不执行 ping：

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=index,codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8556/src_in
```

## 3. 板端最小推流 demo

保存为 `rtsp_demo.py`。输入从 `8556` 读取，输出发布到板端 MediaMTX 的
`8554`；RK3588 使用 `h264_rkmpp`：

```python
import cv2
import subprocess

INPUT_URL = "rtsp://127.0.0.1:8556/src_in"
OUTPUT_URL = "rtsp://127.0.0.1:8554/ai_out"

cap = cv2.VideoCapture(INPUT_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("读取 RTSP 失败，请检查 MediaMTX、FFmpeg 推流和 SSH 隧道")

height, width = frame.shape[:2]
ffmpeg = subprocess.Popen([
    "ffmpeg", "-f", "rawvideo", "-pix_fmt", "bgr24",
    "-s", f"{width}x{height}", "-r", "25", "-i", "-",
    "-an", "-c:v", "h264_rkmpp", "-pix_fmt", "yuv420p",
    "-f", "rtsp", "-rtsp_transport", "tcp", OUTPUT_URL,
], stdin=subprocess.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 在这里插入 YOLO 推理，并将检测框绘制到 frame。
    ffmpeg.stdin.write(frame.tobytes())

cap.release()
ffmpeg.stdin.close()
ffmpeg.wait()
```

运行：

```bash
python3 rtsp_demo.py
```

### 3.1 板端验证输出

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8554/ai_out
```

### 3.2 主机查看输出

主机通过 SSH `-L` 映射后的 `8557` 查看：

```powershell
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8557/ai_out
```

无图形界面时：

```powershell
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8557/ai_out
```

## 重要注意事项

1. RK3588 输入固定为 `127.0.0.1:8556`，输出固定为 `127.0.0.1:8554`。
2. 主机查看输出固定为 `127.0.0.1:8557`。
3. SSH、主机 MediaMTX、主机 FFmpeg、板端 MediaMTX 和推理程序的窗口都不能关闭。
4. VLAN 隔离下 ping 失败是正常现象，不要通过添加路由修复。
