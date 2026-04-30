#!/usr/bin/env python3
"""
K线形态YOLO训练脚本 - 支持CPU/GPU/云端GitHub Actions
用法: python train_kline.py [--epochs 200] [--imgsz 640]
"""

import argparse
import shutil
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    # 检测运行环境
    is_github = os.getenv("GITHUB_ACTIONS") == "true"
    is_cpu_only = shutil.which("nvidia-smi") is None

    print(f"运行模式: GitHub Actions={is_github}, CPU Only={is_cpu_only}")

    # 加载预训练权重
    model = YOLO("yolov8n.pt")
    print("✓ 预训练权重加载成功")

    # 训练配置
    train_args = dict(
        data="src/dataset.yaml",  # 统一路径: src/dataset.yaml
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=30,
        batch=8 if not is_cpu_only else 2,
        device=0 if not is_cpu_only else "cpu",
        workers=4 if not is_cpu_only else 2,
        project="runs/detect",
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,             # K线不能上下翻转
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        erasing=0.4,
        crop_fraction=1.0,
        verbose=True,
        seed=42,
    )

    print(f"开始训练: {args.epochs}轮, 图片尺寸{args.imgsz}px")
    print(f"关键参数: batch={train_args['batch']}, device={'GPU' if not is_cpu_only else 'CPU'}")
    print(f"数据增强: mosaic=1.0, mixup=0.1, hsv增强已开启")

    results = model.train(**train_args)

    # 输出最终评分
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        print("\n" + "="*50)
        print("📊 最终训练结果")
        print("="*50)
        print(f"  mAP50:    {rd.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP50-95: {rd.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {rd.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall:   {rd.get('metrics/recall(B)', 0):.4f}")
        print("="*50)

        map50 = rd.get('metrics/mAP50(B)', 0)
        prec = rd.get('metrics/precision(B)', 0)
        rec = rd.get('metrics/recall(B)', 0)

        score = prec * 0.7 + rec * 0.3
        print(f"\n🎯 实盘可用度得分: {score:.2f}/100")
        if score >= 87:
            print("✅ 优秀，可全自动化运行")
        elif score >= 78:
            print("✅ 良好，可半自动化运行")
        else:
            print("⚠️ 待优化，建议增加标注样本+调整参数")

    # 保存权重到 weights/ 目录（供发布步骤使用）
    best_path = "runs/detect/train/weights/best.pt"
    if os.path.exists(best_path):
        os.makedirs("weights", exist_ok=True)
        shutil.copy(best_path, "weights/best.pt")
        print(f"\n✓ 最佳权重已保存: weights/best.pt")
    else:
        print("\n⚠️ 权重文件未找到，跳过保存")

if __name__ == "__main__":
    main()
