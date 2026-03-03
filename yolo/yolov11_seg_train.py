from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m-seg.pt")  # load a pretrained model

# Train the model
results = model.train(data="./datasets/bolts_dataset/dataset.yaml", epochs=200, imgsz=640, batch=24)