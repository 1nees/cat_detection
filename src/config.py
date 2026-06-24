from pathlib import Path

# ──────────────────────────────────────────────
# ROBOFLOW / DATASET
# ──────────────────────────────────────────────

API_KEY      = "eivXhHMtMfygEMz58GfH"
WORKSPACE    = "miljanas-workspace"
PROJECT_NAME = "is_mackeee_nadji"
VERSION      = 3

DATA_YAML   = "is_mackeee_nadji-3/data.yaml"
IMAGES_VAL  = "is_mackeee_nadji-3/valid/images"
IMAGES_TEST = "is_mackeee_nadji-3/test/images"

# ──────────────────────────────────────────────
# MODEL 
# ──────────────────────────────────────────────

PRETRAINED  = "yolov8n.yaml"
RUN_NAME    = "cat_detection-3"
PROJECT_DIR = "runs/train"

BEST_MODEL = f"runs/detect/{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"

# ──────────────────────────────────────────────
# TRENING — hiperparametri
# ──────────────────────────────────────────────

EPOCHS      = 300
IMGSZ       = 640
BATCH       = 16
PATIENCE    = 30
SEED        = 42
SAVE_PERIOD = 5   

# Augmentacije 
MOSAIC     = 0.5  
COPY_PASTE = 0.3
SCALE      = 0.6
MIXUP      = 0.1
HSV_V      = 0.4
TRANSLATE  = 0.1

# Loss tezine
BOX_GAIN = 9.0
CLS_GAIN = 0.3

# ──────────────────────────────────────────────
# INFERENCIJA / NMS 
# ──────────────────────────────────────────────

CONF_THRESHOLD = 0.5
IOU_NMS        = 0.47
MAX_DET        = 300
IOU_MATCH_THRESHOLD = 0.4

# ──────────────────────────────────────────────
# Broj validacionih slika po epohi u sh
# ──────────────────────────────────────────────

SNAPSHOT_COUNT = 4

# ──────────────────────────────────────────────
# Pomocna funkcija 
# ──────────────────────────────────────────────

def find_latest_best_model() -> str:
    search_root = Path("runs/detect") / PROJECT_DIR
    if not search_root.exists():
        return BEST_MODEL
    candidates = sorted(
        search_root.glob(f"{RUN_NAME}*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else BEST_MODEL
