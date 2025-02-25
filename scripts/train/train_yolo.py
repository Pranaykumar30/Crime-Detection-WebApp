from ultralytics import YOLO
import os

def train_yolo():
    model = YOLO("yolov8n.pt")  # Load the YOLOv8 model
    model.train(
        data="data.yaml",  # Path to dataset configuration file
        epochs=20,         # Number of training epochs
        imgsz=640,         # Image size for training
        batch=8,           # Batch size
        name="yolo",       # Experiment name
        project="models",  # Save directory
        device="cpu",      # Use CPU for training
        verbose=True       # Display training logs
    )

if __name__ == "__main__":
    os.makedirs("models/yolo", exist_ok=True)
    train_yolo()
