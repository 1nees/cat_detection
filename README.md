# Cat Detection

YOLOv8 pipeline za detekciju macaka na slikama. Projekat obuhvata dataset, konfiguraciju, trening, validaciju i evaluaciju.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple) ![PyTorch](https://img.shields.io/badge/PyTorch-deep%20learning-orange)

## Brzi Start

```bash
git clone https://github.com/1nees/cat_detection.git
cd cat_detection
uv sync
uv run python src/train.py
```

Evaluacija:

```bash
uv run python src/evaluate.py
```

Statistika dataseta:

```bash
uv run python src/dataset.py
```

Detekcija na novim slikama:

```bash
uv run python src/inference.py
```

## Struktura Projekta

```
cat_detection/
├── src/
│   ├── config.py           ← hiperparametri
│   ├── dataset.py          ← statistika i vizualizacija dataseta
│   ├── model.py             ← kreiranje i ucitavanje YOLO modela
│   ├── train.py             ← trening + validacija
│   ├── evaluate.py          ← evaluacija i grafici
│   ├── inference.py         ← detekcija na novim slikama
│   └── analysis.py          ← dodatna analiza rezultata
├── data/                    ← YOLO dataset
├── new_images/              ← slike za inference
├── notebooks/                ← eksplorativna analiza
├── docs/                     ← dokumentacija
├── runs/                     ← rezultati treninga i evaluacije
├── pyproject.toml
└── README.md
```

## Dataset

Dataset je sopstveni, slike su samostalno prikupljene, a anotacija je radjena u Roboflow-u. Slike su rasporedjene u train/validation/test skup i eksportovane u YOLO formatu.

```yaml
path: data
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['macka']
```

Podela dataseta:

| Skup | Slike |
|---|---|
| Train | 1569 |
| Validation | 112 |
| Test | 112 |

Pravila anotacije:

- svaki objekat (macka) je oznacen bounding box-om
- box obuhvata vidljivi deo macke
- slike bez objekta se koriste kao background primeri

Kvalitet dataseta se proverava u `train.py` pre treninga: provera postojanja slika i labela, da li svaka slika ima odgovarajuci `.txt` fajl i da li su YOLO koordinate normalizovane u opsegu od 0 do 1.

## Trening

Glavna podesavanja su u `src/config.py`.

Koriste se:

```python
model_yaml = "yolov8n.yaml"
epochs = 300
imgsz = 640
batch = 16
patience = 30
```

Trening je zaustavljen early stopping-om na **168. epohi** (od maksimalno 300), pošto se rezultati na validaciji više nisu poboljšavali.

## Evaluacija

Koriste se standardne metrike za detekciju objekata: mAP50, mAP50-95, Precision, Recall, F1.

Rezultati najboljeg modela (`best.pt`):

| Skup | mAP50 | mAP50-95 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Validation | 0.9156 | 0.6114 | 0.8320 | 0.8527 | 0.8422 |
| Test | 0.9028 | 0.5795 | 0.8463 | 0.8571 | 0.8517 |

Model dobro detektuje macke i na validation i na test skupu, sa konzistentnim rezultatima izmedju ta dva skupa.

Na validation skupu, u `validation_errors.txt` pronadjeno je **24 slike sa greškama**:

- 15 promašenih macaka (model nije detektovao objekat koji postoji)
- 9 viška detekcija (lažne detekcije macke koja ne postoji)

Najčešća greška je manjak jedne detekcije po slici, dok je jedna slika imala 4 promašene macke odjednom (slika sa više macaka na gomili). Lažne detekcije su uglavnom imale nizak confidence (0.50–0.66), što ukazuje da bi viša granica pouzdanosti pri predikciji mogla smanjiti broj viška detekcija.

Rezultati treninga i najbolji model čuvaju se u:

```
runs/detect/runs/train/cat_detection-3-8/
```

Najvazniji fajlovi za pregled:

```
runs/detect/runs/train/cat_detection-3-8/results.png
runs/detect/runs/train/cat_detection-3-8/confusion_matrix.png
runs/detect/runs/train/cat_detection-3-8/training_summary.txt
runs/detect/runs/train/cat_detection-3-8/validation_errors.txt
runs/detect/runs/train/cat_detection-3-8/weights/best.pt
```

## Izvori

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [PyTorch](https://pytorch.org/)
- [Roboflow](https://roboflow.com/)