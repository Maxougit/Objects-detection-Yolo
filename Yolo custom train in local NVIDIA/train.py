from ultralytics import YOLO

def train_yolo():
    model = YOLO("yolov8x.yaml")
    result = model.train(data="data.yaml", epochs=1000)
    return result

if __name__ == '__main__':
    train_yolo()
