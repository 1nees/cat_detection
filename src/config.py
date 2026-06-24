"""
config.py
─────────────────────────────────────────────
Centralizovana konfiguracija za ceo projekat (train.py, evaluate.py, inference.py).

Sve tri skripte importuju vrednosti odavde, tako da se nista ne duplira i
ne moze se desiti da se npr. putanja do modela ili IoU prag promene na
jednom mestu a zaborave na drugom.

Upotreba u drugim fajlovima:
    from config import BEST_MODEL, IMGSZ, IOU_NMS, ...
"""

from pathlib import Path

# ──────────────────────────────────────────────
# ROBOFLOW / DATASET
# ──────────────────────────────────────────────

API_KEY      = "eivXhHMtMfygEMz58GfH"
WORKSPACE    = "miljanas-workspace"
PROJECT_NAME = "is_mackeee_nadji"
VERSION      = 3

# Putanje do podataka (koristi evaluate.py; train.py ih dobija direktno od Roboflow-a)
DATA_YAML   = "is_mackeee_nadji-3/data.yaml"
IMAGES_VAL  = "is_mackeee_nadji-3/valid/images"
IMAGES_TEST = "is_mackeee_nadji-3/test/images"

# ──────────────────────────────────────────────
# MODEL — naziv i gde se cuva/trazi
# ──────────────────────────────────────────────

PRETRAINED  = "yolov8n.yaml"   # arhitektura od nule (ne .pt) — zahtev projekta
RUN_NAME    = "cat_detection-2"
PROJECT_DIR = "runs/train"

# Putanja do najboljeg istreniranog modela.
# VAZNO: ovo mora da se poklapa sa stvarnim folderom koji Ultralytics napravi
# (runs/train/<RUN_NAME>/weights/best.pt — Ultralytics sam dodaje broj na kraju
#  ako folder vec postoji, npr. cat_detection2, cat_detection3...).
# Posle svakog treninga provericate stvarnu putanju u terminalu i azurirajte ovde.
BEST_MODEL = f"runs/detect/{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"

# ──────────────────────────────────────────────
# TRENING — hiperparametri (koristi samo train.py)
# ──────────────────────────────────────────────

EPOCHS      = 300
IMGSZ       = 640
BATCH       = 16
PATIENCE    = 40
SEED        = 42
SAVE_PERIOD = 5     # cuvaj checkpoint svakih N epoha (za epoch snapshots)

# Augmentacije — povecavaju raznovrsnost scena sa zbijenim/preklapajucim mackama
MOSAIC     = 1.0    # spaja 4 slike u jednu -> vise maca odjednom u jednom prikazu
COPY_PASTE = 0.3    # lepi izrezane objekte iz drugih slika -> vise preklapanja
SCALE      = 0.6    # agresivnije zumiranje/odzumiranje -> vise variacije velicine
MIXUP      = 0.1    # blago mesanje dve slike, pomaze generalizaciji

# Augmentacije koje pomazu kod kamuflaze (npr. macka u travi) i promene konteksta
HSV_V      = 0.4    # nasumicna promena osvetljenja (brightness) -> model ne uci samo
                     # na jednoj svetlini scene (mace u senci/suncu izgledaju drugacije)
TRANSLATE  = 0.1    # nasumicno pomeranje slike (do 10% sirine/visine) -> macka nije
                     # uvek centrirana, model uci da je trazi i na rubovima kadra

# Loss tezine — vise box (lokalizacija), manje cls (samo 1 klasa = trivijalna klasifikacija)
BOX_GAIN = 9.0
CLS_GAIN = 0.3

# ──────────────────────────────────────────────
# INFERENCIJA / NMS — koriste sva tri fajla (train, evaluate, inference)
# ──────────────────────────────────────────────

CONF_THRESHOLD = 0.25   # prag pouzdanosti — ispod ovoga se detekcija odbacuje
IOU_NMS        = 0.4    # NMS prag (Non-Max Suppression) — vise = tolerantniji
                         # prema preklapajucim box-ovima (mace blizu jedna drugoj)
MAX_DET        = 300    # gornja granica broja detekcija po slici

# IoU prag za RUCNO poklapanje GT <-> predikcija u analizi gresaka (train.py),
# NIJE isto sto i IOU_NMS — ovo je prag "da li se predikcija dovoljno poklapa
# sa stvarnom mackom da se racuna kao pogodak" u nasoj sopstvenoj analizi.
IOU_MATCH_THRESHOLD = 0.45

# Koliko validacionih slika prikazati po epohi u snapshotima (train.py)
SNAPSHOT_COUNT = 4

# ──────────────────────────────────────────────
# Pomocna funkcija — ako zelis da automatski nadje najnoviji "cat_detection*" folder
# umesto da rucno azuriras BEST_MODEL svaki put
# ──────────────────────────────────────────────

def find_latest_best_model() -> str:
    """Pronalazi najnoviji runs/detect/.../cat_detection*/weights/best.pt po datumu izmene.
    Korisno kad Ultralytics napravi cat_detection2, cat_detection3... pri ponovnom treningu."""
    search_root = Path("runs/detect") / PROJECT_DIR
    if not search_root.exists():
        return BEST_MODEL
    candidates = sorted(
        search_root.glob(f"{RUN_NAME}*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else BEST_MODEL
