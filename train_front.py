from ultralytics import YOLO
import os

def train_cnic_model():
    # Load pretrained model
    model = YOLO('yolov8n.pt')  # Good balance for document detection
    
    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=15,
        save=True,
        device='cpu',
        workers=4,
        optimizer='auto',
        lr0=0.01,
        augment=True
    )
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    train_cnic_model()