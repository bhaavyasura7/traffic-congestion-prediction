from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # or yolov8s.pt

model.train(
    data='data.yaml', 
    epochs=50,
    imgsz=640,
    batch=8
)