from pathlib import Path

import torch
from PIL import Image, ImageDraw
from roboflow import Roboflow
from ultralytics import YOLO

# ──────────────────────────────────────────────
# KONFIGURACIJA — menjaj ovde
# ──────────────────────────────────────────────

API_KEY        = "eivXhHMtMfygEMz58GfH"
WORKSPACE      = "miljanas-workspace"
PROJECT_NAME   = "is_mackeee_nadji"
VERSION        = 3

EPOCHS         = 50
IMGSZ          = 640
BATCH          = 16
PATIENCE       = 10
SEED           = 42
SAVE_PERIOD    = 5          # čuvaj checkpoint svakih N epoha (za epoch snapshots)
PRETRAINED     = "yolov8n.pt"
RUN_NAME       = "cat_detection"
PROJECT_DIR    = "runs/train"

# Koliko validacionih slika prikazati po epohi u snapshotima
SNAPSHOT_COUNT = 4
CONF_THRESHOLD = 0.25       # prag za analizu grešaka
IOU_THRESHOLD  = 0.45       # IoU prag za matching GT i predikcija

# ──────────────────────────────────────────────
# POMOĆNE FUNKCIJE
# ──────────────────────────────────────────────

def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_images(folder: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)


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


def box_iou(a: list, b: list) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def detection_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0


# ──────────────────────────────────────────────
# PROVERA DATASETA
# ──────────────────────────────────────────────

def check_dataset(data_yaml: Path) -> None:
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text())
    class_count = int(cfg.get("nc", len(cfg.get("names", []))))
    errors = []
    base = data_yaml.parent

    for split in ("train", "val", "test"):
        if split not in cfg:
            errors.append(f"Split '{split}' ne postoji u data.yaml.")
            continue
        images_dir = (base / cfg[split].replace("..", str(base))).resolve()
        labels_dir = (base / cfg[split].replace("..", str(base)).replace("images", "labels")).resolve()
        print(f"  Trazim slike za '{split}' u: {images_dir}")

        if not images_dir.exists():
            errors.append(f"Nema foldera sa slikama za '{split}': {images_dir}")
            continue
        if not labels_dir.exists():
            errors.append(f"Nema foldera sa labelama za '{split}': {labels_dir}")
            continue

        image_paths = list_images(images_dir)
        if not image_paths:
            errors.append(f"Split '{split}' nema nijednu sliku.")

        for img_path in image_paths:
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                errors.append(f"Nedostaje label fajl: {lbl_path}")
                continue
            for i, line in enumerate(lbl_path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"Los format u {lbl_path}:{i}")
                    continue
                try:
                    cid = int(float(parts[0]))
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    errors.append(f"Labela nije broj u {lbl_path}:{i}")
                    continue
                if not 0 <= cid < class_count:
                    errors.append(f"Nepostojeca klasa {cid} u {lbl_path}:{i}")
                if any(v < 0 or v > 1 for v in coords):
                    errors.append(f"Koordinate nisu normalizovane u {lbl_path}:{i}")

    if errors:
        preview = "\n".join(f"  - {e}" for e in errors[:20])
        extra = f"\n  ... i jos {len(errors) - 20} problema." if len(errors) > 20 else ""
        raise ValueError(f"Provera dataseta nije prosla:\n{preview}{extra}")

    print(f"✓ Dataset provera prosla — klase: {class_count}, splitovi: train/valid/test")


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
        weights_path = Path(getattr(trainer, "last", Path(trainer.save_dir) / "weights" / "last.pt"))
        if not weights_path.exists():
            return
        snap_model = YOLO(str(weights_path))
        save_epoch_snapshot(snap_model, epoch, val_images, labels_dir, save_dir, device)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


# ──────────────────────────────────────────────
# ANALIZA GREŠAKA
# ──────────────────────────────────────────────

def analyze_errors(model: YOLO, image_paths: list[Path], labels_dir: Path, device: str, save_path: Path) -> None:
    print("\nAnaliziram greske na validacionom skupu...")
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        device=device,
        verbose=False,
    )

    report = []
    total_errors = 0
    images_with_errors = 0

    for img_path, result in zip(image_paths, results):
        with Image.open(img_path) as im:
            w, h = im.size
        labels = read_labels(labels_dir / f"{img_path.stem}.txt", w, h)
        preds = []
        if result.boxes is not None:
            for box, cls, conf in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                preds.append({"class_id": int(cls), "box": box, "conf": conf})

        matched = set()
        image_errors = []

        for gt in labels:
            best_i, best_iou = None, 0
            for i, pred in enumerate(preds):
                if i in matched:
                    continue
                iou = box_iou(gt["box"], pred["box"])
                if iou > best_iou:
                    best_iou, best_i = iou, i

            gt_name = model.names.get(gt["class_id"], str(gt["class_id"]))

            if best_i is None or best_iou < IOU_THRESHOLD:
                image_errors.append(f"  - PROMASENO: ocekivano '{gt_name}'")
            else:
                matched.add(best_i)
                pred_name = model.names.get(preds[best_i]["class_id"], str(preds[best_i]["class_id"]))
                if preds[best_i]["class_id"] != gt["class_id"]:
                    image_errors.append(
                        f"  - POGRESNA KLASA: ocekivano '{gt_name}', dobijeno '{pred_name}' "
                        f"(conf={preds[best_i]['conf']:.2f}, IoU={best_iou:.2f})"
                    )

        for i, pred in enumerate(preds):
            if i not in matched:
                pred_name = model.names.get(pred["class_id"], str(pred["class_id"]))
                image_errors.append(f"  - VISAK: lazna detekcija '{pred_name}' (conf={pred['conf']:.2f})")

        if image_errors:
            total_errors += len(image_errors)
            images_with_errors += 1
            report.append(str(img_path.name))
            report.extend(image_errors)
            report.append("")

    save_path.write_text("\n".join(report) if report else "Nema pronadjenih gresaka.\n")
    print(f"  Ukupno gresaka: {total_errors} na {images_with_errors} slika")
    print(f"  Izvestaj sacuvan: {save_path}")


# ──────────────────────────────────────────────
# TRAINING SUMMARY
# ──────────────────────────────────────────────

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


def write_summary(save_dir: Path, val_metrics, test_metrics, device: str) -> None:
    def metric_block(title, m):
        p, r = m.box.mp, m.box.mr
        return [
            f"{title}:",
            f"  mAP50:    {m.box.map50:.4f}",
            f"  mAP50-95: {m.box.map:.4f}",
            f"  Precision:{p:.4f}",
            f"  Recall:   {r:.4f}",
            f"  F1:       {detection_f1(p, r):.4f}",
        ]

    lines = [
        "═══ TRAINING SUMMARY ═══",
        "",
        f"Uredjaj:  {device}",
        f"Model:    {PRETRAINED}",
        f"Epohe:    {EPOCHS}",
        f"Imgsz:    {IMGSZ}",
        f"Batch:    {BATCH}",
        f"Patience: {PATIENCE}",
        "",
        *metric_block("Validacija (best.pt)", val_metrics),
        "",
        *metric_block("Test (best.pt)", test_metrics),
    ]

    summary_path = save_dir / "training_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary sacuvan: {summary_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main() -> None:
    device = resolve_device()
    print(f"Uredjaj: {device}")

    # 1. Preuzmi dataset
    print("\nPreuzimam dataset sa Roboflowa...")
    rf = Roboflow(api_key=API_KEY)
    dataset = rf.workspace(WORKSPACE).project(PROJECT_NAME).version(VERSION).download("yolov8")
    data_yaml = Path(dataset.location) / "data.yaml"

    # 2. Proveri dataset
    print("\nProveravam dataset...")
    check_dataset(data_yaml)

    # 3. Pripremi val slike za snapshots
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text())
    base = data_yaml.parent
    val_images_dir = (base / cfg["val"].replace("..", str(base))).resolve()
    val_labels_dir = (base / cfg["val"].replace("..", str(base)).replace("images", "labels")).resolve()
    val_images = list_images(val_images_dir)

    # 4. Trening
    print(f"\nPokrecem trening ({EPOCHS} epoha)...")
    model = YOLO(PRETRAINED)

    # Snapshot folder — znaćemo ga tek posle treninga, privremeno koristimo placeholder
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
        workers=0,
        project=PROJECT_DIR,
        name=RUN_NAME,
    )

    save_dir   = Path(results.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    last_model = save_dir / "weights" / "last.pt"
    print(f"\nTrening zavrsen. Rezultati: {save_dir}")
    print(f"Najbolji model: {best_model}")

    # 5. Validacija
    print("\nPokrecem validaciju (best.pt)...")
    best = YOLO(str(best_model))
    val_metrics = best.val(
        data=str(data_yaml),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
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
        project=PROJECT_DIR,
        name=f"{RUN_NAME}_test",
        plots=True,
    )
    print_metrics("TEST", test_metrics)

    # 7. Training summary
    write_summary(save_dir, val_metrics, test_metrics, device)

    # 8. Analiza grešaka
    analyze_errors(best, val_images, val_labels_dir, device, save_dir / "validation_errors.txt")

    print("\nSve zavrseno!")
    print(f"Epoch snapshots: {save_dir / 'epoch_snapshots'}")
    print(f"Greske: {save_dir / 'validation_errors.txt'}")
    print(f"Summary: {save_dir / 'training_summary.txt'}")


if __name__ == "__main__":
    main()