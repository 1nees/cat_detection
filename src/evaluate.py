from pathlib import Path

from ultralytics import YOLO

from config import (
    BEST_MODEL, DATA_YAML, IMAGES_VAL, IMAGES_TEST,
    IMGSZ, BATCH, PROJECT_DIR, RUN_NAME, IOU_NMS,
)
from analysis import print_metrics, plot_training_curves, plot_confusion_matrix, plot_predictions

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main() -> None:
    model_path = Path(BEST_MODEL)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model nije pronadjen: {model_path}\n"
            "Proveri putanju u BEST_MODEL varijabli u config.py."
        )

    data_yaml = Path(DATA_YAML)
    run_dir   = model_path.parent.parent   # runs/train/cat_detection/

    model = YOLO(str(model_path))

    print("\nPokrecem evaluaciju na validacionom skupu...")
    val_metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        workers=0,
        iou=IOU_NMS,
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
        iou=IOU_NMS,
        plots=True,
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_eval_test",
    )
    print_metrics("TEST", test_metrics)

    print("\nCrtam grafik treninga...")
    plot_training_curves(run_dir)

    print("\nPrikazujem matricu konfuzije...")
    plot_confusion_matrix(run_dir)

    print("\nPrikazujem predikcije na test slikama...")
    plot_predictions(model, Path(IMAGES_TEST), n=6, title="test")

    print("\nPrikazujem predikcije na validacionim slikama...")
    plot_predictions(model, Path(IMAGES_VAL), n=6, title="val")


if __name__ == "__main__":
    main()
