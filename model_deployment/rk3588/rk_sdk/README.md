# RK runtime staging directory

This replaces the AX `msp_sdk` staging point.  Board images normally provide
RKNN Runtime and RGA under `/usr`; for an isolated/cross build, stage only the
matching target SDK files here:

```text
rk_sdk/
  include/rknn_api.h
  lib/aarch64/librknnrt.so
```

The files must come from the same RKNN Toolkit2 release installed on the
RK3588.  MPP/RGA/FFmpeg/OpenCV are resolved from the target sysroot because
their ABI must match the board image.
