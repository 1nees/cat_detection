from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────
# KONFIGURACIJA — uskladi sa train.py
# ──────────────────────────────────────────────

BEST_MODEL = "runs/detect/runs/train/cat_detection-6/weights/best.pt"
DATA_YAML  = "is_mackeee_nadji-3/data.yaml"
IMAGES_VAL = "is_mackeee_nadji-3/valid/images"
IMAGES_TEST = "is_mackeee_nadji-3/test/images"
IMGSZ       = 640
BATCH       = 16
PROJECT_DIR = "runs/train"
RUN_NAME    = "cat_detection"

# ──────────────────────────────────────────────
# POMOĆNE FUNKCIJE
# ──────────────────────────────────────────────

def detection_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0


def print_metrics(title: str, metrics) -> None:
    p, r = metrics.box.mp, metrics.box.mr
    print(f"\n{'─'*40}")
    print(f" {title}")
    print(f"{'─'*40}")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision:{p:.4f}")
    print(f"  Recall:   {r:.4f}")
    print(f"  F1:       {detection_f1(p, r):.4f}")
    print(f"{'─'*40}")


# ──────────────────────────────────────────────
# GRAFIK — loss i metrike kroz epohe
# ──────────────────────────────────────────────

def plot_training_curves(run_dir: Path) -> None:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        print(f"Nema results.csv u {run_dir}, preskacam grafik treninga.")
        return

    import pandas as pd
    history = pd.read_csv(results_csv)
    history.columns = history.columns.str.strip()
    epochs = history["epoch"] + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train/box_loss"], label="train box loss")
    ax1.plot(epochs, history["val/box_loss"],   label="val box loss")
    ax1.plot(epochs, history["train/cls_loss"], label="train cls loss")
    ax1.plot(epochs, history["val/cls_loss"],   label="val cls loss")
    ax1.set_xlabel("Epoha")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss kroz epohe")
    ax1.legend(fontsize=8)

    ax2.plot(epochs, history["metrics/mAP50(B)"],    label="mAP50")
    ax2.plot(epochs, history["metrics/mAP50-95(B)"], label="mAP50-95")
    ax2.plot(epochs, history["metrics/precision(B)"], label="Precision")
    ax2.plot(epochs, history["metrics/recall(B)"],    label="Recall")
    ax2.set_xlabel("Epoha")
    ax2.set_ylabel("Vrednost")
    ax2.set_title("Metrike kroz epohe")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = run_dir / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Grafik treninga sacuvan: {out}")


# ──────────────────────────────────────────────
# GRAFIK — matrica konfuzije
# ──────────────────────────────────────────────

def plot_confusion_matrix(run_dir: Path) -> None:
    cm_path = run_dir / "confusion_matrix.png"
    if not cm_path.exists():
        print(f"Nema confusion_matrix.png u {run_dir}, preskacam.")
        return

    img = plt.imread(str(cm_path))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Matrica konfuzije")
    plt.tight_layout()
    plt.show()
    print(f"  Matrica konfuzije ucitana iz: {cm_path}")


# ──────────────────────────────────────────────
# GRAFIK — predikcije na test slikama
# ──────────────────────────────────────────────

def plot_predictions(model: YOLO, images_dir: Path, n: int = 6, seed: int = 0, title: str = "") -> None:
    if not images_dir.exists():
        print(f"Nema foldera sa slikama: {images_dir}")
        return

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in extensions)

    if not image_paths:
        print(f"Nema slika u {images_dir}")
        return

    rng = np.random.default_rng(seed)
    selected = rng.choice(image_paths, size=min(n, len(image_paths)), replace=False)

    results = model.predict(
        source=[str(p) for p in selected],
        imgsz=IMGSZ,
        conf=0.25,
        verbose=False,
    )

    cols = min(3, len(results))
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.atleast_1d(axes).flatten()

    for ax, result in zip(axes, results):
        ax.imshow(result.plot()[..., ::-1])
        ax.axis("off")
        ax.set_title(Path(result.path).name, fontsize=8)

    for ax in axes[len(results):]:
        ax.axis("off")

    plt.suptitle(f"Predikcije — {title}", fontsize=12)
    plt.tight_layout()

    out = Path(PROJECT_DIR) / RUN_NAME / f"predictions_{title}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Predikcije sacuvane: {out}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main() -> None:
    model_path = Path(BEST_MODEL)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model nije pronadjen: {model_path}\n"
            "Proveri putanju u BEST_MODEL varijabli na vrhu fajla."
        )

    data_yaml = Path(DATA_YAML)
    run_dir   = model_path.parent.parent   # runs/train/cat_detection/

    model = YOLO(str(model_path))

    # 1. Evaluacija na validacionom skupu
    print("\nPokrecem evaluaciju na validacionom skupu...")
    val_metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        workers=0,
        plots=True,
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_eval_val",
    )
    print_metrics("VALIDACIJA", val_metrics)

    # 2. Evaluacija na test skupu
    print("\nPokrecem evaluaciju na test skupu...")
    test_metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        workers=0,
        plots=True,
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_eval_test",
    )
    print_metrics("TEST", test_metrics)

    # 3. Grafik loss i metrika kroz epohe
    print("\nCrtam grafik treninga...")
    plot_training_curves(run_dir)

    # 4. Matrica konfuzije
    print("\nPrikazujem matricu konfuzije...")
    plot_confusion_matrix(run_dir)

    # 5. Predikcije na test slikama
    print("\nPrikazujem predikcije na test slikama...")
    plot_predictions(model, Path(IMAGES_TEST), n=6, title="test")

    # 6. Predikcije na val slikama
    print("\nPrikazujem predikcije na validacionim slikama...")
    plot_predictions(model, Path(IMAGES_VAL), n=6, title="val")


if __name__ == "__main__":
    main()