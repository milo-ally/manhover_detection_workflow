import time
import json
import os
import sys

"""
# 全量编译:
    cmake -S . -B build -DOpenCV_DIR=/opt/opencv-rk3588/lib/cmake/opencv4 -DBUILD_TESTING=OFF # 指定 OpenCV 路径
    cmake --build build
    ./bin/demo_helmet --help
# 添加新模型:
    cmake -S . -B build -DOpenCV_DIR=/opt/opencv-rk3588/lib/cmake/opencv4 -DBUILD_TESTING=OFF # 指定 OpenCV 路径
    cmake --build build --target <plugin> demo_helmet
    ./bin/demo_helmet --help
# 启动 MediaMTX:
    ./mediamtx ./mediamtx.yml
# 运行并推流:
    ./bin/demo_helmet -c config/streams_config.json --mediamtx 127.0.0.1:8000
# 播放第 11 路:
    rtsp://127.0.0.1:8554/live11
"""

# 模擬配置路徑 (與後端一致)
CONFIG_PATH = "/dev/shm/ai_config.json"

# 默認參數
current_config = {
    "conf_thres": 0.5,
    "nms_thres": 0.45,
    "model_name": "default"
}

def load_config():
    """嘗試讀取中控下發的配置"""
    global current_config
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                new_config = json.load(f)
                # 簡單比較是否有變化
                if new_config != current_config:
                    print(f"\n[SYSTEM] Config Update Detected!")
                    print(f"Old: {current_config}")
                    print(f"New: {new_config}")
                    # 在這裡執行實際的模型參數設置 API，例如:
                    # model.set_threshold(new_config['conf_thres'])
                    current_config.update(new_config)
        except Exception as e:
            print(f"Config read error: {e}")

def main():
    print("RK3588 AI Demo Started...")
    print(f"Listening for config at: {CONFIG_PATH}")
    
    # 模擬推理循環
    frame_count = 0
    while True:
        # 1. 每一幀都檢查配置 (或每 10 幀檢查一次以節省 IO)
        if frame_count % 10 == 0:
            load_config()
        
        # 2. 模擬推理
        # print(f"Inferencing with Model: {current_config['model_name']} | Conf: {current_config['conf_thres']}", end='\r')
        
        # 3. 模擬 RTSP 推流（實際程序使用 RK3588 MPP/RKNN 管線）
        # push_rtsp_frame(frame)
        
        frame_count += 1
        time.sleep(0.033) # 模擬 30 FPS

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
