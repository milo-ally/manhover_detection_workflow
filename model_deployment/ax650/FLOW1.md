# AX650 基础推流 FLOW1‑SSH‑TUNNEL

本文档是 AX650 的基础 RTSP 输入 / 输出链路（SSH 端口转发隧道版本）。AX650 侧的实际井盖 AI 流程序
请使用同目录 `manhole_cover_detection` 小工程和 `FLOW2.md`；本文件中的
Python 代码只用于验证主机输入流和最小推流链路。

环境前提:

- 主机可 SSH 登录边缘设备 `root@172.19.30.3`

- 主机与边缘设备存在 VLAN 隔离，**无法互相 ping 通**，依靠 SSH 端口转发透传 TCP RTSP 流量

- SSH 隧道会话窗口一旦关闭，全部端口映射立即失效

## 1 主机安装 \(MediaMTX \+ FFMpeg\)

> 主机为 Windows 平台，使用 Windows 二进制包
> 
> 

### 11 安装 FFmpeg

Windows 下载 ffmpeg release 二进制，配置系统环境变量，保证 PowerShell/CMD 可直接调用`ffmpeg`。

校验是否安装成功

```powershell
ffmpeg --version
ffmpeg -encoders | findstr libx264 # 出现 libx264 代表编码器正常
```

### 12 安装 MediaMTX

下载 Windows amd64 版本 release 二进制包，省去编译。

解压包内直接包含 `mediamtx.exe` 和 `mediamtx.yml`。MediaMTX v1.10.0 使用 `protocols`
配置 RTSP 传输方式；将配置文件修改：

```yml
protocols: [tcp]
rtspAddress: :554
```

`paths` 保持默认的 `all_others:` 配置即可，不要改成文档旧版本中的
`paths: all:`。

### 13 主机防火墙放行 554 端口

Windows 防火墙入站规则，放行 `554/TCP`。

### 14 启动 MediaMTX

```powershell
.\mediamtx.exe mediamtx.yml
```

看到日志： `RTSP listener opened on :554`，启动成功。这个终端窗口不要关闭。

### 15 主机推流 `test.mp4`

> VLAN 隔离，**不能使用主机局域网 IP 给板子访问**，仅本机 \[127001\]\(127001\) 使用。
> 
> 

```powershell
ffmpeg -re -stream_loop -1 -i ./test.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:554/src_in
```

如果当前目录不是 `mediamtx.exe` 和 `test.mp4` 所在目录，请使用绝对路径，或先切换工作目录。

### 16 主机本地验证输入流

```powershell
ffplay rtsp://127.0.0.1:554/src_in -rtsp_transport tcp
```

可以正常播放视频，代表上游链路没问题。无图形界面时可使用无窗口校验：

```powershell
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:554/src_in
```

### 17 建立 SSH 双向端口转发隧道（关键步骤）

新开独立 PowerShell 窗口执行，**该窗口全程不能关闭，关闭隧道失效**

```powershell
ssh -R 8556:127.0.0.1:554 -L 8557:127.0.0.1:8554 root@172.19.30.3
```

- `-R 8556:127.0.0.1:554`：远程转发，Windows 本机 554 映射到板子 `127.0.0.1:8556`，板子读取输入流

- `-L 8557:127.0.0.1:8554`：本地转发，板子 8554 映射回 Windows 本机`127.0.0.1:8557`，主机查看 AI 输出流

> 登录成功且无 `remote port forwarding failed` 警告即隧道就绪。
> 边缘设备读取输入流地址：`rtsp://127.0.0.1:8556/src_in`。
> 主机上 MediaMTX、FFmpeg 推流、SSH 隧道终端都不要关闭。
> 
> 

## 2 边缘设备

### 21 安装 ffmpeg

```bash
sudo apt update
sudo apt install ffmpeg -y
```

### 22 边缘设备部署 MediaMTX

下载 linux\_aarch64 二进制包，板子本地 MediaMTX 用于输出 AI 处理后流，监听 8554 端口。

```bash
wget https://github.com/bluenviron/mediamtx/releases/download/v1.10.0/mediamtx_v1.10.0_linux_arm64.tar.gz
tar -xvzf mediamtx_v1.10.0_linux_arm64.tar.gz
```

修改 mediamtxyml

```yml
protocols: [tcp]
```

启动板子侧 mediamtx

```bash
./mediamtx mediamtx.yml
```

### 23 设备网络 \& 流校验

> VLAN 隔离，**ping HOST\_IP 会失败，属于正常现象，不需要 ping 通**。
> 通过 SSH 隧道本地回环地址做 ffprobe 校验。
> 
> 

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=index,codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8556/src_in
# 预计输出:
## index=0
## codec_name=hevc
## width=1138
## height=720
## index=1
## codec_name=aac
```

### 24 边缘设备 Python 最小 demo

保存下面脚本为 `rtsp_demo.py`：

```python
import cv2
import subprocess

INPUT_URL = "rtsp://127.0.0.1:8556/src_in"
OUTPUT_URL = "rtsp://127.0.0.1:8554/ai_out"

cap = cv2.VideoCapture(INPUT_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 先读到第一帧再拿分辨率
ret, frame = cap.read()
if not ret:
    print("读取RTSP流失败，请检查ssh隧道、mediamtx、Windows推流")
    cap.release()
    exit(1)

height, width = frame.shape[:2]

# 使用系统ffmpeg，编码器mpeg4（内置，不需要libx264），删除preset/tune
ffmpeg = subprocess.Popen([
    "ffmpeg", "-f", "rawvideo", "-pix_fmt", "bgr24",
    "-s", f"{width}x{height}", "-r", "25", "-i", "-",
    "-an", "-c:v", "mpeg4", "-b:v", "4000k",
    "-f", "rtsp", "-rtsp_transport", "tcp", OUTPUT_URL,
], stdin=subprocess.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取流失败")
        break

    # 在这里插入 YOLO推理，绘制检测框到frame
    ffmpeg.stdin.write(frame.tobytes())

cap.release()
ffmpeg.stdin.close()
ffmpeg.wait()
```

运行：

```bash
python3 rtsp_demo.py
```

### 25 边缘设备校验输出流

```bash
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8554/ai_out
```

## 3 主机验证 AI 输出画面（借助 SSH‑L 本地转发）

Windows 主机执行，访问映射回本机的端口 8557：

```powershell
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8557/ai_out
```

无图形界面时使用 ffprobe 验证输出流：

```powershell
ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 rtsp://127.0.0.1:8557/ai_out
```

## 重要注意事项

1. 本方案仅转发 TCP 流量，ICMP ping 完全不可用，不影响 RTSP 业务。

2. SSH 隧道窗口关闭，所有端口映射全部失效。

3. 板子禁止执行带`via`的 ip‑route 路由命令，会直接断开 SSH 会话，无串口无法恢复。

4. 若后续使用网线直连主机以太网口与 AX650 网口，则可切回原版 FLOW1，使用真实 IP 互相访问。

> (Note: May contain AI-generated content.)
