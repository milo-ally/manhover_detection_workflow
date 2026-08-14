from ultralytics import YOLO


def main():
    # YOLO11s 比 YOLO11n 精度更好，同时模型规模还适合后续部署。
    model = YOLO("yolo11s.pt")
    model.train(
        # 数据与输出
        data="/home/milo/workspace/LYG_manhover_detection_workflow/model_train/Manhole-Cover-5Class-3/data.yaml",
        project="/home/milo/workspace/LYG_manhover_detection_workflow/model_train/runs",
        name="manhole-cover-yolo11s-production",
        exist_ok=True,
        pretrained=True,

        # 基础训练参数
        epochs=300,       # 最大训练轮数，实际会受 patience 早停控制
        patience=100,     # 验证集较小、波动大，早停等待时间设长一点
        batch=8,          # 训练高峰期主要调整batch, 降低显存压力
        imgsz=640,        # 根据实际情况调整
        device=0,
        workers=8,
        seed=42,
        deterministic=True,
        amp=True,

        # 优化器与学习率
        optimizer="AdamW",
        lr0=0.0003,       # 学习率保守一些，保证微调稳定
        lrf=0.05,         # 最终学习率 = lr0 * lrf
        momentum=0.937,
        weight_decay=0.001,
        warmup_epochs=8.0,
        cos_lr=True,

        # 损失权重
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # 数据增强
        mosaic=0.2,       # 弱化 mosaic，避免小数据集增强过强
        close_mosaic=80,  # 最后 80 轮关闭 mosaic，让模型贴近真实图像
        degrees=0.0,
        translate=0.03,
        scale=0.2,
        fliplr=0.5,
        hsv_h=0.008,
        hsv_s=0.3,
        hsv_v=0.2,

        # 验证与保存
        val=True,
        plots=True,
        save=True,
        save_period=20, # epoch*.pt 用于回滚和对比；如果只想保留 best.pt/last.pt，改成 -1。
    )


if __name__ == "__main__":
    main()
