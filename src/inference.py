from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/train/cat_detection-3/weights/best.pt"
)

results = model.predict(
    source="new_images",
    save=True,
    conf=0.25
)