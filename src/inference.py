from pathlib import Path
from ultralytics import YOLO

from config import BEST_MODEL, CONF_THRESHOLD, IOU_NMS, MAX_DET

# Napomena: svi parametri se nalaze u config.py — menjaj ih tamo,
# ne ovde, da budu uskladjeni sa train.py i evaluate.py.

model = YOLO(BEST_MODEL)

results = model.predict(
    source="new_images",    # ← folder sa novim slikama
    save=True,
    conf=CONF_THRESHOLD,
    iou=IOU_NMS,         # NMS prag — veci broj = tolerantniji na preklapajuce box-ove (mace blizu jedna drugoj)
    agnostic_nms=False,  # NMS racunaj posebno po klasi (ovde je samo 1 klasa, ali korektno za buducnost)
    max_det=MAX_DET,     # gornja granica detekcija po slici
    workers=0
)