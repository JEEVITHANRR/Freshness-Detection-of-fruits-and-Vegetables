"""
ResNet18-based Freshness Detection Model

Uses a fine-tuned ResNet18 for binary classification:
- Fresh
- Rotten

This provides more accurate freshness detection than basic classifiers.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Model path
MODEL_PATH = "models/resnet18_freshness.pth"

# Image preprocessing
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Class labels
CLASSES = ["Fresh", "Rotten"]

# Global model
_model = None

def load_model():
    """Load the ResNet18 freshness model."""
    global _model
    
    if _model is not None:
        return _model
    
    # Create ResNet18 model
    model = models.resnet18(weights=None)
    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Load weights if available
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            print("✅ ResNet18 freshness model loaded")
        except Exception as e:
            print(f"⚠️ Error loading model weights: {e}")
            # Use pretrained weights for feature extraction
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(num_features, 2)
    else:
        print("⚠️ No trained model found. Using pretrained ResNet18")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(num_features, 2)
    
    model.eval()
    _model = model
    return model

def predict_freshness(image_path):
    """
    Predict freshness of a food item.
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict: {
            "label": "Fresh" or "Rotten",
            "confidence": 0.95,
            "probabilities": {"Fresh": 0.95, "Rotten": 0.05}
        }
    """
    model = load_model()
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_tensor = TRANSFORM(image).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        # Get prediction
        confidence, predicted_idx = torch.max(probabilities, 0)
        label = CLASSES[predicted_idx.item()]
        
        return {
            "label": label,
            "confidence": confidence.item(),
            "probabilities": {
                "Fresh": probabilities[0].item(),
                "Rotten": probabilities[1].item()
            }
        }
        
    except Exception as e:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "error": str(e)
        }

def predict_freshness_batch(image_paths):
    """Predict freshness for multiple images."""
    return [predict_freshness(path) for path in image_paths]


# Training function
def train_resnet18_freshness(dataset_path, epochs=20, batch_size=32, lr=0.001):
    """
    Train ResNet18 for freshness classification.
    
    Dataset structure:
        dataset_path/
            train/
                fresh/
                rotten/
            val/
                fresh/
                rotten/
    """
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    print("=" * 60)
    print("🚀 RESNET18 FRESHNESS TRAINING")
    print("=" * 60)
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = TRANSFORM
    
    # Load datasets
    train_dataset = datasets.ImageFolder(f"{dataset_path}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{dataset_path}/val", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Classes: {train_dataset.classes}")
    
    # Create model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze early layers
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    
    # Modify final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 2)
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    best_acc = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"\n🔧 Training on: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
        
        train_acc = train_correct / len(train_dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        
        val_acc = val_correct / len(val_dataset)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  💾 Saved best model (Val Acc: {val_acc:.4f})")
    
    print("\n" + "=" * 60)
    print(f"✅ Training Complete! Best Val Accuracy: {best_acc:.4f}")
    print(f"💾 Model saved to: {MODEL_PATH}")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    # Train the model
    dataset_path = "dataset/processed"
    train_resnet18_freshness(dataset_path, epochs=20)
