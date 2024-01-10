from ultralytics import YOLO

model = YOLO("yolov8m.yaml")

result = model.train(data="data.yaml", epochs=20)