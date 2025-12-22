"""
Item-Specific Detection Model Trainer

Trains a YOLOv8 detection model to identify EXACT food items:
- apple, banana, tomato, carrot, etc.

This provides more accurate item identification than YOLO-World's
general-purpose detection.
"""

from ultralytics import YOLO
import os
import json
from pathlib import Path

# Configuration
DETECTION_DATASET = "dataset/detection"
MODEL_OUTPUT = "models/item_detector"

# Training Configuration
TRAINING_CONFIG = {
    "model": "yolov8m.pt",        # Medium detection model
    "epochs": 100,                 # More epochs for detection
    "imgsz": 640,                  # Standard detection size
    "batch": 16,                   # Batch size
    "patience": 15,                # Early stopping patience
    "lr0": 0.01,                   # Initial LR
    "lrf": 0.01,                   # Final LR
    "warmup_epochs": 5,            # Warmup
    "cos_lr": True,                # Cosine scheduler
    "amp": True,                   # Mixed precision
    # Augmentation
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.1,
}

# Classes for detection
FOOD_CLASSES = [
    "apple", "banana", "orange", "pear", "grape", "strawberry", 
    "mango", "kiwi", "tomato", "potato", "cucumber", "carrot", 
    "broccoli", "cauliflower", "cabbage", "onion", "eggplant", 
    "corn", "mushroom", "chili"
]

def create_dataset_yaml():
    """Create YOLO format dataset configuration."""
    dataset_path = os.path.abspath(DETECTION_DATASET)
    
    yaml_content = f"""
# Food Item Detection Dataset
path: {dataset_path}
train: images/train
val: images/val

# Classes
names:
"""
    for i, cls in enumerate(FOOD_CLASSES):
        yaml_content += f"  {i}: {cls}\n"
    
    yaml_path = f"{DETECTION_DATASET}/data.yaml"
    os.makedirs(DETECTION_DATASET, exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created dataset config: {yaml_path}")
    return yaml_path

def setup_detection_dataset():
    """Create directory structure for detection dataset."""
    dirs = [
        f"{DETECTION_DATASET}/images/train",
        f"{DETECTION_DATASET}/images/val",
        f"{DETECTION_DATASET}/labels/train",
        f"{DETECTION_DATASET}/labels/val",
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("📁 Detection dataset structure created")
    print("\n⚠️ To train the item detector, you need labeled data:")
    print("   1. Add images to: dataset/detection/images/train/")
    print("   2. Add YOLO format labels to: dataset/detection/labels/train/")
    print("   3. Format: <class_id> <x_center> <y_center> <width> <height>")
    print("\n💡 Tip: Use tools like LabelImg or Roboflow to create annotations")

def check_existing_model():
    """Check if we have an existing trained model."""
    model_path = f"{MODEL_OUTPUT}/weights/best.pt"
    if os.path.exists(model_path):
        print(f"✅ Existing model found: {model_path}")
        return True
    return False

def train_detector():
    """Train the item detection model."""
    print("=" * 60)
    print("🍎 ITEM DETECTION MODEL TRAINER")
    print("=" * 60)
    
    # Setup
    setup_detection_dataset()
    yaml_path = create_dataset_yaml()
    
    # Check for training data
    train_images = f"{DETECTION_DATASET}/images/train"
    if not os.path.exists(train_images) or len(os.listdir(train_images)) == 0:
        print("\n❌ No training images found!")
        print("   Please add labeled images to the dataset first.")
        print("\n📝 Quick Start Guide:")
        print("   1. Download a food detection dataset (Kaggle, Roboflow)")
        print("   2. Place images in: dataset/detection/images/train/")
        print("   3. Place labels in: dataset/detection/labels/train/")
        print("   4. Re-run this script")
        return
    
    # Count images
    num_images = len([f for f in os.listdir(train_images) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"\n📊 Found {num_images} training images")
    
    if num_images < 50:
        print("⚠️ Warning: Very small dataset. Consider adding more images.")
    
    # Load model
    print(f"\n📦 Loading base model: {TRAINING_CONFIG['model']}")
    model = YOLO(TRAINING_CONFIG["model"])
    
    # Train
    print("\n🏋️ Starting training...")
    results = model.train(
        data=yaml_path,
        epochs=TRAINING_CONFIG["epochs"],
        imgsz=TRAINING_CONFIG["imgsz"],
        batch=TRAINING_CONFIG["batch"],
        patience=TRAINING_CONFIG["patience"],
        lr0=TRAINING_CONFIG["lr0"],
        lrf=TRAINING_CONFIG["lrf"],
        warmup_epochs=TRAINING_CONFIG["warmup_epochs"],
        cos_lr=TRAINING_CONFIG["cos_lr"],
        amp=TRAINING_CONFIG["amp"],
        hsv_h=TRAINING_CONFIG["hsv_h"],
        hsv_s=TRAINING_CONFIG["hsv_s"],
        hsv_v=TRAINING_CONFIG["hsv_v"],
        degrees=TRAINING_CONFIG["degrees"],
        translate=TRAINING_CONFIG["translate"],
        scale=TRAINING_CONFIG["scale"],
        fliplr=TRAINING_CONFIG["fliplr"],
        mosaic=TRAINING_CONFIG["mosaic"],
        mixup=TRAINING_CONFIG["mixup"],
        project="models",
        name="item_detector",
        exist_ok=True,
        verbose=True
    )
    
    # Validate
    print("\n📊 Validating...")
    metrics = model.val()
    print(f"\n✅ Training Complete!")
    print(f"   mAP50: {metrics.box.map50:.2%}")
    print(f"   mAP50-95: {metrics.box.map:.2%}")
    
    print("\n" + "=" * 60)
    print(f"🎉 Model saved to: {MODEL_OUTPUT}/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    train_detector()
