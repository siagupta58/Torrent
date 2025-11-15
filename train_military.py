from ultralytics import YOLO

print("Starting training on military object dataset.")

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # nano (fastest)
# or use: 'yolov8s.pt' (small), 'yolov8m.pt' (medium), 'yolov8l.pt' (large)

# Train on your military dataset
results = model.train(
    data='C:/Users/siagu/Torrent/military_dataset.yaml',
    epochs=20,  # adjust based on your needs
    imgsz=320,
    batch=8,  # lower this if you get memory errors
    name='military_model',
    device='cpu',  # change to 0 if you have NVIDIA GPU
    patience=10,  # early stopping
    save=True,
    project='military_training'
)

print("Training complete!")
print(f"Best model saved to: military_training/military_model/weights/best.pt")

# Validate the model
metrics = model.val()
print(f"\nmAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")