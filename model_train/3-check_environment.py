import platform
import subprocess
import sys


print(f"python: {sys.version.split()[0]} ({sys.executable})")
print(f"platform: {platform.platform()}")

try:
    import ultralytics

    print(f"ultralytics: {ultralytics.__version__}")
except Exception as exc:
    print(f"ultralytics: not available ({exc})")

try:
    import torch

    print(f"torch: {torch.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"gpu {i}: {props.name}, {props.total_memory / 1024**3:.2f} GB")
except Exception as exc:
    print(f"torch: not available ({exc})")

try:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    print("nvidia-smi:")
    print((result.stdout or result.stderr).strip())
except Exception as exc:
    print(f"nvidia-smi: not available ({exc})")
