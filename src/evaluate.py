from ultralytics import YOLO

def main():
    model = YOLO(
        "runs/detect/runs/train/cat_detection-3/weights/best.pt"
    )

    metrics = model.val(workers=0)

    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"Precision: {metrics.box.mp}")
    print(f"Recall: {metrics.box.mr}")

if __name__ == "__main__":
    main()