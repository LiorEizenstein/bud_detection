from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='treex-dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    project='runs',
    name='bud_detector',
    patience=20,
)
