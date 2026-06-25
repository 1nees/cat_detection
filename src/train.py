from pathlib import Path

import torch
from PIL import Image, ImageDraw
from roboflow import Roboflow
from ultralytics import YOLO
import glob

from config import (
    API_KEY, WORKSPACE, PROJECT_NAME, VERSION,
    EPOCHS, IMGSZ, BATCH, PATIENCE, SEED, SAVE_PERIOD,
    PRETRAINED, RUN_NAME, PROJECT_DIR,
    MOSAIC, COPY_PASTE, SCALE, MIXUP, HSV_V, TRANSLATE, BOX_GAIN, CLS_GAIN,
    CONF_THRESHOLD, IOU_NMS, IOU_MATCH_THRESHOLD, SNAPSHOT_COUNT,
)
from analysis import print_metrics, write_summary, analyze_errors
from dataset import check_dataset, list_images, resolve_split_dir, generate_dataset_report

# ──────────────────────────────────────────────
# POMOĆNE FUNKCIJE
# ──────────────────────────────────────────────

def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def class_color(class_id: int) -> tuple[int, int, int]:
    colors = [(220, 56, 56), (45, 126, 222), (48, 168, 84), (230, 145, 56), (140, 80, 200)]
    return colors[class_id % len(colors)]


def draw_box(draw: ImageDraw.ImageDraw, box: list, label: str, color: tuple) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    tb = draw.textbbox((x1, y1), label)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    y_text = max(0, y1 - th - 6)
    draw.rectangle((x1, y_text, x1 + tw + 8, y_text + th + 6), fill=color)
    draw.text((x1 + 4, y_text + 3), label, fill="white")


def yolo_to_xyxy(values: list, w: int, h: int) -> list:
    cx, cy, bw, bh = values
    return [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]


def read_labels(label_path: Path, w: int, h: int) -> list[dict]:
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        labels.append({
            "class_id": int(float(parts[0])),
            "box": yolo_to_xyxy([float(v) for v in parts[1:5]], w, h),
        })
    return labels

# ──────────────────────────────────────────────
# EPOCH VALIDATION SNAPSHOTS
# ──────────────────────────────────────────────

def render_panel(image_path: Path, boxes: list[dict], names: dict, title: str, panel_w: int) -> Image.Image:
    with Image.open(image_path) as src:
        img = src.convert("RGB")
    scale = panel_w / img.width
    img = img.resize((panel_w, max(1, int(img.height * scale))))
    scaled_boxes = [{**b, "box": [v * scale for v in b["box"]]} for b in boxes]

    title_h = 30
    panel = Image.new("RGB", (img.width, img.height + title_h), "white")
    panel.paste(img, (0, title_h))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), title, fill="black")

    for item in scaled_boxes:
        name = names.get(item["class_id"], str(item["class_id"]))
        label = f"{name} {item['conf']:.2f}" if "conf" in item else name
        shifted = [item["box"][0], item["box"][1] + title_h, item["box"][2], item["box"][3] + title_h]
        draw_box(draw, shifted, label, class_color(item["class_id"]))

    return panel


def save_epoch_snapshot(model: YOLO, epoch: int, val_images: list[Path], labels_dir: Path, save_dir: Path, device: str) -> None:
    images_to_show = val_images[:SNAPSHOT_COUNT]
    results = model.predict(
        source=[str(p) for p in images_to_show],
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        device=device,
        verbose=False,
    )

    rows = []
    for img_path, result in zip(images_to_show, results):
        with Image.open(img_path) as im:
            w, h = im.size
        gt_boxes = read_labels(labels_dir / f"{img_path.stem}.txt", w, h)
        pred_boxes = []
        if result.boxes is not None:
            for box, cls, conf in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                pred_boxes.append({"class_id": int(cls), "box": box, "conf": conf})

        left  = render_panel(img_path, gt_boxes,   model.names, f"GT | {img_path.name}", IMGSZ)
        right = render_panel(img_path, pred_boxes, model.names, f"Predikcija | epoha {epoch}", IMGSZ)
        row = Image.new("RGB", (left.width + right.width + 10, max(left.height, right.height)), "white")
        row.paste(left, (0, 0))
        row.paste(right, (left.width + 10, 0))
        rows.append(row)

    if not rows:
        return

    padding = 10
    combined_w = max(r.width for r in rows)
    combined_h = sum(r.height for r in rows) + padding * (len(rows) - 1)
    combined = Image.new("RGB", (combined_w, combined_h), "white")
    y = 0
    for row in rows:
        combined.paste(row, (0, y))
        y += row.height + padding

    out_path = save_dir / "epoch_snapshots" / f"epoch_{epoch:03d}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path, quality=95)
    print(f"  → Snapshot sacuvan: {out_path.name}")


def add_snapshot_callback(model: YOLO, val_images: list[Path], labels_dir: Path, save_dir: Path, device: str) -> None:
    def on_fit_epoch_end(trainer):
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        if epoch != 1 and epoch % SAVE_PERIOD != 0:
            return
        weights_path = Path(getattr(trainer, "last", Path(trainer.save_dir) / "weights" / "last.pt"))
        if not weights_path.exists():
            return
        snap_model = YOLO(str(weights_path))
        save_epoch_snapshot(snap_model, epoch, val_images, labels_dir, save_dir, device)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main() -> None:
    device = resolve_device()
    print(f"Uredjaj: {device}")

    print("\nPreuzimam dataset sa Roboflowa...")
    rf = Roboflow(api_key=API_KEY)
    dataset = rf.workspace(WORKSPACE).project(PROJECT_NAME).version(VERSION).download("yolov8")
    data_yaml = Path(dataset.location) / "data.yaml"

    print("\nProveravam dataset...")
    check_dataset(data_yaml)

    print("\nGenerisem izvestaj o datasetu...")
    generate_dataset_report(data_yaml, output_dir=Path(PROJECT_DIR) / RUN_NAME / "dataset_report")

    import yaml
    cfg = yaml.safe_load(data_yaml.read_text())
    base = data_yaml.parent
    val_images_dir = resolve_split_dir(base, cfg["val"])
    val_labels_dir = Path(str(val_images_dir).replace("images", "labels"))
    val_images = list_images(val_images_dir)

    print(f"\nPokrecem trening ({EPOCHS} epoha, patience={PATIENCE})...")
    model = YOLO(PRETRAINED)

    snapshot_save_dir = Path(PROJECT_DIR) / RUN_NAME
    add_snapshot_callback(model, val_images, val_labels_dir, snapshot_save_dir, device)

    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        patience=PATIENCE,
        seed=SEED,
        save_period=SAVE_PERIOD,
        workers=8,
        project=PROJECT_DIR,
        name=RUN_NAME,
        mosaic=MOSAIC,           
        copy_paste=COPY_PASTE,   
        scale=SCALE,             
        mixup=MIXUP,             
        hsv_v=HSV_V,         
        translate=TRANSLATE, 
        box=BOX_GAIN,      
        cls=CLS_GAIN,          
        iou=IOU_NMS,           
    )

    save_dir   = Path(results.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    last_model = save_dir / "weights" / "last.pt"

    actual_epochs = int(getattr(model.trainer, "epoch", EPOCHS - 1)) + 1

    print(f"\nTrening zavrsen. Rezultati: {save_dir}")
    print(f"Najbolji model: {best_model}")

    # koliko epoha je trening zapravo trajao, da li je doslo do early stoppinga

    if actual_epochs < EPOCHS:
        print(f"⚠ Early stopping aktiviran posle {actual_epochs}/{EPOCHS} epoha "
              f"(nema poboljsanja {PATIENCE} epoha zaredom).")

    # 5. Validacija
    print("\nPokrecem validaciju (best.pt)...")
    best = YOLO(str(best_model))
    val_metrics = best.val(
        data=str(data_yaml),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        iou=IOU_NMS,
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_validation",
        plots=True,
    )
    print_metrics("VALIDACIJA", val_metrics)

    # 6. Test evaluacija
    print("\nPokrecem test evaluaciju (best.pt)...")
    test_metrics = best.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        iou=IOU_NMS,
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_test",
        plots=True,
    )
    print_metrics("TEST", test_metrics)

    write_summary(save_dir, val_metrics, test_metrics, device, actual_epochs)

    analyze_errors(best, val_images, val_labels_dir, device, save_dir / "validation_errors.txt", read_labels)

    print("\nSve zavrseno!")
    
    # Obrisi yolo cache fajlove
    for f in glob.glob("yolo*.pt"):
        Path(f).unlink(missing_ok=True)

    print(f"Epoch snapshots: {save_dir / 'epoch_snapshots'}")
    print(f"Greske: {save_dir / 'validation_errors.txt'}")
    print(f"Summary: {save_dir / 'training_summary.txt'}")


if __name__ == "__main__":
    main()