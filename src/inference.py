from pathlib import Path
from ultralytics import YOLO

BEST_MODEL = "runs/detect/runs/train/cat_detection-6/weights/best.pt"

model = YOLO(BEST_MODEL)

results = model.predict(
    source="new_images",    # ← folder sa novim slikama
    save=True,
    conf=0.25,
    workers=0
)