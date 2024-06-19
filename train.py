from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n-seg.yaml")  # build a new model from YAML
    model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="data.yaml", epochs=10, imgsz=640)  # '0' refers to the first GPU
