"""
analysis.py
─────────────────────────────────────────────
Sve funkcije vezane za ANALIZU USPEHA modela: metrike, racunanje IoU-a,
analiza gresaka (promaseno/visak/pogresan broj objekata), pisanje summary-ja
i crtanje grafika (loss/metrike kroz epohe, matrica konfuzije, predikcije).

Koristi je i train.py (na kraju treninga) i evaluate.py (samostalna evaluacija
veec istreniranog modela) — tako da se ne duplira kod i da su definicije
metrika garantovano iste na oba mesta.

Upotreba:
    from analysis import print_metrics, analyze_errors, write_summary, ...
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from config import IMGSZ, BATCH, PATIENCE, EPOCHS, PRETRAINED, CONF_THRESHOLD, IOU_NMS, IOU_MATCH_THRESHOLD, PROJECT_DIR, RUN_NAME


# ──────────────────────────────────────────────
# OSNOVNE METRIKE
# ──────────────────────────────────────────────

def detection_f1(precision: float, recall: float) -> float:
    """F1 skor iz precision i recall — harmonijska sredina, balansira oba."""
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0


def box_iou(a: list, b: list) -> float:
    """IoU (Intersection over Union) dva bounding box-a u xyxy formatu.
    Koristi se za RUCNO poklapanje GT <-> predikcija u analizi gresaka
    (razlikuje se od IOU_NMS koji kontrolise NMS prag u Ultralytics-u)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def print_metrics(title: str, metrics) -> None:
    """Ispisuje standardne YOLO metrike (mAP50, mAP50-95, Precision, Recall, F1)
    za rezultat koji vraca model.val()."""
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
# ANALIZA GREŠAKA (promaseno / pogresna klasa / visak / pogresan broj)
# ──────────────────────────────────────────────

def analyze_errors(model: YOLO, image_paths: list, labels_dir: Path, device: str, save_path: Path,
                    read_labels_fn) -> None:
    """Pokrece predikciju na svim image_paths i poredi sa GT labelama.
    Belezi 4 tipa gresaka: PROMASENO (GT bez para), POGRESNA KLASA, VISAK
    (lazna detekcija), i BROJ MACAKA (kad se broj GT i broj predikcija ne poklapaju
    — najvazniji pokazatelj problema sa zbijenim/preklapajucim objektima).

    read_labels_fn: funkcija read_labels(label_path, w, h) -> list[dict], prosledjena
    iz train.py da ne moramo da duplujemo parsiranje YOLO label fajlova ovde."""
    print("\nAnaliziram greske na validacionom skupu...")
    from PIL import Image

    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        iou=IOU_NMS,
        device=device,
        verbose=False,
    )

    report = []
    total_errors = 0
    images_with_errors = 0

    # Metrika specificna za "blizu jedna drugoj" / preklapajuce objekte:
    # da li se BROJ predikcija poklapa sa BROJEM GT objekata po slici (bez obzira na IoU/klasu)
    count_mismatch_images = 0
    count_diffs = []  # (pred_count - gt_count) po slici, za analizu da li model sistematski potcenjuje broj

    for img_path, result in zip(image_paths, results):
        with Image.open(img_path) as im:
            w, h = im.size
        labels = read_labels_fn(labels_dir / f"{img_path.stem}.txt", w, h)
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

            if best_i is None or best_iou < IOU_MATCH_THRESHOLD:
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

        gt_count, pred_count = len(labels), len(preds)
        if gt_count != pred_count:
            count_mismatch_images += 1
            count_diffs.append(pred_count - gt_count)
            image_errors.append(
                f"  - BROJ MACAKA: GT={gt_count}, predikcija={pred_count} "
                f"({'manjak' if pred_count < gt_count else 'visak'} {abs(pred_count - gt_count)})"
            )

        if image_errors:
            total_errors += len(image_errors)
            images_with_errors += 1
            report.append(str(img_path.name))
            report.extend(image_errors)
            report.append("")

    save_path.write_text("\n".join(report) if report else "Nema pronadjenih gresaka.\n", encoding="utf-8")
    print(f"  Ukupno gresaka: {total_errors} na {images_with_errors} slika")
    print(f"  Izvestaj sacuvan: {save_path}")

    # ── Sumarna statistika za problem "pogresan broj objekata na slici" ──
    n_images = len(image_paths)
    if n_images:
        avg_diff = sum(count_diffs) / len(count_diffs) if count_diffs else 0.0
        n_undercount = sum(1 for d in count_diffs if d < 0)  # model je prebrojao MANJE maca nego ih ima (manjak)
        n_overcount  = sum(1 for d in count_diffs if d > 0)  # model je prebrojao VISE maca nego ih ima (visak)
        print(f"\n  ── Tacnost brojanja maca po slici ──")
        print(f"  Slika sa pogresnim brojem: {count_mismatch_images}/{n_images} "
              f"({100 * count_mismatch_images / n_images:.1f}%)")
        print(f"  Od toga - manjak (NMS/preklapanje sumnja): {n_undercount}, visak (lazne detekcije): {n_overcount}")
        print(f"  Prosecna razlika (predikcija - GT): {avg_diff:+.2f}")
        if n_undercount > n_overcount:
            print("  → Model sistematski PROPUSTA macke (vise manjkova nego viskova) — "
                  "ovo je tipican znak NMS-a koji brise preklapajuce box-ove kod zbijenih maca. "
                  "Probaj povecati 'iou' prag pri predikciji/treningu i dodati copy_paste/mosaic augmentaciju.")


# ──────────────────────────────────────────────
# TRAINING SUMMARY
# ──────────────────────────────────────────────

def write_summary(save_dir: Path, val_metrics, test_metrics, device: str, actual_epochs: int) -> None:
    """Pise training_summary.txt sa hiperparametrima i finalnim metrikama
    (validacija + test) za brz pregled rezultata jednog treninga."""
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

    early_stopped = actual_epochs < EPOCHS

    lines = [
        "═══ TRAINING SUMMARY ═══",
        "",
        f"Uredjaj:  {device}",
        f"Model:    {PRETRAINED}",
        f"Epohe (max):   {EPOCHS}",
        f"Epohe (zapravo): {actual_epochs}" + ("  (zaustavljeno early stopping-om)" if early_stopped else ""),
        f"Imgsz:    {IMGSZ}",
        f"Batch:    {BATCH}",
        f"Patience: {PATIENCE}",
        "",
        *metric_block("Validacija (best.pt)", val_metrics),
        "",
        *metric_block("Test (best.pt)", test_metrics),
    ]

    summary_path = save_dir / "training_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary sacuvan: {summary_path}")


# ──────────────────────────────────────────────
# GRAFIK — loss i metrike kroz epohe
# ──────────────────────────────────────────────

def plot_training_curves(run_dir: Path) -> None:
    """Crta i cuva grafik loss-a (box/cls kroz epohe) i metrika (mAP50,
    mAP50-95, Precision, Recall) na osnovu results.csv koji Ultralytics
    automatski generise tokom treninga."""
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
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Grafik treninga sacuvan: {out}")


# ──────────────────────────────────────────────
# GRAFIK — matrica konfuzije
# ──────────────────────────────────────────────

def plot_confusion_matrix(run_dir: Path) -> None:
    """Prikazuje confusion_matrix.png koji Ultralytics automatski generise
    pri model.val(plots=True)."""
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
# GRAFIK — predikcije na test/val slikama
# ──────────────────────────────────────────────

def plot_predictions(model: YOLO, images_dir: Path, n: int = 6, seed: int = 0, title: str = "") -> None:
    """Nasumicno bira n slika iz images_dir, pokrece predikciju i prikazuje
    ih sa nacrtanim bounding box-ovima (model.plot()) u gridu."""
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
        conf=CONF_THRESHOLD,
        iou=IOU_NMS,
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
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Predikcije sacuvane: {out}")