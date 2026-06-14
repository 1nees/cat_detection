from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="eivXhHMtMfygEMz58GfH")
project = rf.workspace("miljanas-workspace").project("is_mackeee_nadji_proba")
version = project.version(1)
dataset = version.download("yolov8")

#ucitavamo yolov8 model
model = YOLO("yolov8n.pt")

#treniramo model
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="cat_detection",
    project="runs/train",
    patience=10,
    device=0,
    workers=0
)
                