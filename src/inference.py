from pathlib import Path
from ultralytics import YOLO

from config import BEST_MODEL, CONF_THRESHOLD, IOU_NMS, MAX_DET

model = YOLO(BEST_MODEL)

results = model.predict(
    source="new_images",
    save=True,
    conf=CONF_THRESHOLD,
    iou=IOU_NMS,         
    agnostic_nms=False,
    max_det=MAX_DET,   
    workers=0
)