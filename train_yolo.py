from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='data.yaml',  # path to your dataset config
    epochs=20,
    imgsz=640,
    batch=16,
    name='yolo_model',
    device=0  # 0 for GPU, 'cpu' for CPU
)

print("Training complete!")