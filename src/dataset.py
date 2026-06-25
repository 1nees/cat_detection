from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from config import DATA_YAML, PROJECT_DIR, RUN_NAME

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ──────────────────────────────────────────────
# UCITAVANJE / PUTANJE
# ──────────────────────────────────────────────

def load_dataset_config(data_yaml: str | Path = DATA_YAML) -> dict:
    return yaml.safe_load(Path(data_yaml).read_text())


def resolve_split_dir(base: Path, split_path: str) -> Path:
    if split_path.startswith("../"):
        split_path = split_path[3:]
    return (base / split_path).resolve()


def split_image_dir(cfg: dict, split: str, data_yaml: str | Path = DATA_YAML) -> Path:
    base = Path(data_yaml).parent
    return resolve_split_dir(base, cfg[split])


def split_label_dir(cfg: dict, split: str, data_yaml: str | Path = DATA_YAML) -> Path:
    images_dir = split_image_dir(cfg, split, data_yaml)
    return Path(str(images_dir).replace("images", "labels"))


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def list_labels(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.txt"))


def get_class_names(cfg: dict) -> list[str]:
    names = cfg["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, x, y, w, h = parts
        boxes.append((int(float(class_id)), float(x), float(y), float(w), float(h)))
    return boxes


# ──────────────────────────────────────────────
# PROVERA ISPRAVNOSTI DATASETA (pre treninga)
# ──────────────────────────────────────────────

def check_dataset(data_yaml: Path) -> None:
    cfg = yaml.safe_load(data_yaml.read_text())
    class_count = int(cfg.get("nc", len(cfg.get("names", []))))
    errors = []
    base = data_yaml.parent

    for split in ("train", "val", "test"):
        if split not in cfg:
            errors.append(f"Split '{split}' ne postoji u data.yaml.")
            continue

        images_dir = resolve_split_dir(base, cfg[split])
        labels_dir = Path(str(images_dir).replace("images", "labels"))
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
# STATISTIKA DATASETA
# ──────────────────────────────────────────────

@dataclass
class SplitStats:
    name: str
    images: int
    labels: int
    objects: int
    class_counts: Counter


def bbox_size_category(area: float) -> str:
    if area < 0.01:
        return "tiny <1%"
    if area < 0.05:
        return "small 1-5%"
    if area < 0.20:
        return "medium 5-20%"
    return "large >20%"


def split_stats(cfg: dict, split: str, data_yaml: str | Path = DATA_YAML) -> SplitStats:
    images_dir = split_image_dir(cfg, split, data_yaml)
    labels_dir = split_label_dir(cfg, split, data_yaml)

    counts = Counter()
    for label_path in list_labels(labels_dir):
        for class_id, *_ in parse_label_file(label_path):
            counts[class_id] += 1

    return SplitStats(
        name=split,
        images=len(list_images(images_dir)),
        labels=len(list_labels(labels_dir)),
        objects=sum(counts.values()),
        class_counts=counts,
    )


def dataset_summary(data_yaml: str | Path = DATA_YAML) -> dict[str, SplitStats]:
    cfg = load_dataset_config(data_yaml)
    return {
        split: split_stats(cfg, split, data_yaml)
        for split in ("train", "val", "test")
        if split in cfg
    }


def print_dataset_summary(data_yaml: str | Path = DATA_YAML) -> None:
    cfg = load_dataset_config(data_yaml)
    class_names = get_class_names(cfg)
    print(f"Dataset: {Path(data_yaml).resolve()}")
    print(f"Klase: {', '.join(class_names)}")
    for stats in dataset_summary(data_yaml).values():
        print(
            f"{stats.name}: slike={stats.images}, label fajlovi={stats.labels}, "
            f"objekti={stats.objects}, raspodela={dict(stats.class_counts)}"
        )


def objects_per_image(cfg: dict, split: str, data_yaml: str | Path = DATA_YAML) -> list[int]:
    images_dir = split_image_dir(cfg, split, data_yaml)
    labels_dir = split_label_dir(cfg, split, data_yaml)
    counts = []
    for img_path in list_images(images_dir):
        label_path = labels_dir / f"{img_path.stem}.txt"
        counts.append(len(parse_label_file(label_path)))
    return counts


def bbox_areas(cfg: dict, split: str, data_yaml: str | Path = DATA_YAML) -> list[float]:
    labels_dir = split_label_dir(cfg, split, data_yaml)
    areas = []
    for label_path in list_labels(labels_dir):
        for _, _, _, w, h in parse_label_file(label_path):
            areas.append(w * h)
    return areas


# ──────────────────────────────────────────────
# GRAFICI
# ──────────────────────────────────────────────

def plot_class_distribution_by_split(summary: dict[str, SplitStats], class_names: list[str]) -> plt.Figure:
    splits = list(summary)
    x = np.arange(len(splits))
    width = 0.8 / max(1, len(class_names))

    fig, ax = plt.subplots(figsize=(7, 4))
    for class_id, name in enumerate(class_names):
        values = [summary[s].class_counts[class_id] for s in splits]
        offset = (class_id - (len(class_names) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name)
        ax.bar_label(bars, padding=3, fontsize=8)

    ax.set_xticks(x, splits)
    ax.set_ylabel("Broj objekata")
    ax.set_title("Raspodela klasa po splitu (train/val/test)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_objects_per_image(cfg: dict, data_yaml: str | Path = DATA_YAML) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        counts = objects_per_image(cfg, split, data_yaml)
        if counts:
            bins = np.arange(max(counts) + 2) - 0.5
            ax.hist(counts, bins=bins, alpha=0.55, label=split)

    ax.set_xlabel("Broj macaka na slici")
    ax.set_ylabel("Broj slika")
    ax.set_title("Distribucija broja objekata po slici")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_bbox_size_categories(cfg: dict, data_yaml: str | Path = DATA_YAML) -> plt.Figure:
    categories = ["tiny <1%", "small 1-5%", "medium 5-20%", "large >20%"]
    splits = [s for s in ("train", "val", "test") if s in cfg]
    counts = {
        s: Counter(bbox_size_category(a) for a in bbox_areas(cfg, s, data_yaml))
        for s in splits
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(splits))
    colors = ["#E45756", "#F58518", "#54A24B", "#4C78A8"]
    for category, color in zip(categories, colors):
        values = np.array([counts[s][category] for s in splits])
        bars = ax.bar(splits, values, bottom=bottom, label=category, color=color)
        ax.bar_label(bars, labels=[str(v) if v else "" for v in values], label_type="center", fontsize=8)
        bottom += values

    ax.set_ylabel("Broj bbox-eva")
    ax.set_title("Velicina bounding box-ova po splitu")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_bbox_area_histogram(cfg: dict, data_yaml: str | Path = DATA_YAML) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        areas = [a * 100 for a in bbox_areas(cfg, split, data_yaml)]
        if areas:
            ax.hist(areas, bins=20, alpha=0.55, label=split)

    ax.set_xlabel("Povrsina bbox-a (% slike)")
    ax.set_ylabel("Broj bbox-eva")
    ax.set_title("Distribucija povrsine bounding box-eva")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def bbox_text_summary(cfg: dict, data_yaml: str | Path = DATA_YAML) -> str:
    class_names = get_class_names(cfg)
    lines = ["BBox summary", ""]

    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        areas = np.array(bbox_areas(cfg, split, data_yaml))
        if len(areas) == 0:
            continue
        category_counts = Counter(bbox_size_category(a) for a in areas)
        obj_counts = objects_per_image(cfg, split, data_yaml)

        lines.extend([
            f"{split}:",
            f"  bbox-evi: {len(areas)}",
            f"  slike: {len(obj_counts)}  (od toga sa 2+ objekta: {sum(1 for c in obj_counts if c >= 2)})",
            f"  area min/median/max: {areas.min()*100:.2f}% / {np.median(areas)*100:.2f}% / {areas.max()*100:.2f}%",
            "  velicine: " + ", ".join(
                f"{c}={category_counts[c]}" for c in ("tiny <1%", "small 1-5%", "medium 5-20%", "large >20%")
            ),
            "",
        ])

    return "\n".join(lines)


# ──────────────────────────────────────────────
# GLAVNI IZVESTAJ 
# ──────────────────────────────────────────────

def generate_dataset_report(data_yaml: str | Path = DATA_YAML,
                             output_dir: str | Path | None = None) -> Path:
    cfg = load_dataset_config(data_yaml)
    class_names = get_class_names(cfg)
    summary = dataset_summary(data_yaml)

    out = Path(output_dir) if output_dir else Path(PROJECT_DIR) / RUN_NAME / "dataset_report"
    out.mkdir(parents=True, exist_ok=True)

    plots = {
        "01_class_distribution_by_split.png": plot_class_distribution_by_split(summary, class_names),
        "02_objects_per_image.png": plot_objects_per_image(cfg, data_yaml),
        "03_bbox_size_categories.png": plot_bbox_size_categories(cfg, data_yaml),
        "04_bbox_area_histogram.png": plot_bbox_area_histogram(cfg, data_yaml),
    }
    for filename, fig in plots.items():
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)

    (out / "bbox_summary.txt").write_text(bbox_text_summary(cfg, data_yaml), encoding="utf-8")
    print(f"Izvestaj o datasetu sacuvan u: {out}")
    return out


if __name__ == "__main__":
    print_dataset_summary()
    report_dir = generate_dataset_report()
    print(f"Grafici sacuvani u: {report_dir}")
