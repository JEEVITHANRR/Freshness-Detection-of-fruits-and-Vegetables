from ultralytics import YOLO
import os

# Model paths
FRESHNESS_MODEL_PATH = "models/freshness_classifier/weights/best.pt"
ITEM_DETECTOR_PATH = "models/item_detector/weights/best.pt"

# Load the trained YOLOv8 Classification Model
freshness_model = None
item_detector = None

def load_models():
    """Load all required models."""
    global freshness_model, item_detector
    
    # Load freshness classifier
    try:
        freshness_model = YOLO(FRESHNESS_MODEL_PATH)
        print("✅ Freshness classifier loaded")
    except Exception as e:
        print(f"⚠️ Freshness model not found: {e}")
        # Try fallback to nano model
        try:
            freshness_model = YOLO("yolov8n-cls.pt")
            print("   Using pretrained fallback")
        except:
            freshness_model = None
    
    # Load custom item detector if available
    if os.path.exists(ITEM_DETECTOR_PATH):
        try:
            item_detector = YOLO(ITEM_DETECTOR_PATH)
            print("✅ Custom item detector loaded")
        except Exception as e:
            print(f"⚠️ Item detector not found: {e}")
            item_detector = None

# Load on import
load_models()

# Confidence calibration thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 80,     # Very confident
    "medium": 60,   # Somewhat confident  
    "low": 40       # Not very confident
}

def get_confidence_level(confidence):
    """Convert raw confidence to human-readable level."""
    if confidence >= CONFIDENCE_THRESHOLDS["high"]:
        return "High"
    elif confidence >= CONFIDENCE_THRESHOLDS["medium"]:
        return "Medium"
    else:
        return "Low"

def predict_freshness_with_confidence(image_path, item_name=None):
    """
    Predicts freshness using the custom trained YOLOv8 model.
    
    Args:
        image_path: Path to the image file
        item_name: Optional item name for context
        
    Returns:
        label (str): "Fresh ✅" or "Rotten ❌" with item name if provided
        confidence (float): Confidence percentage (0-100)
    """
    if freshness_model is None:
        return "Model Error", 0.0

    try:
        # Run inference with optimized settings
        results = freshness_model(
            image_path, 
            verbose=False,
            imgsz=320  # Match training size
        )
        
        # Extract top result
        probs = results[0].probs
        top1_index = probs.top1
        confidence = probs.top1conf.item() * 100  # Convert to percentage
        
        # Get class name from names dict
        label = results[0].names[top1_index]
        
        # Determine freshness status
        is_fresh = "fresh" in label.lower()
        
        # Create display label
        if item_name:
            # Include item name for exact identification
            item_display = item_name.title()
            if is_fresh:
                display_label = f"Fresh {item_display} ✅"
            else:
                display_label = f"Rotten {item_display} ❌"
        else:
            if is_fresh:
                display_label = "Fresh ✅"
            else:
                display_label = "Rotten ❌"
        
        # Add confidence level indicator
        conf_level = get_confidence_level(confidence)
        
        return display_label, round(confidence, 2)
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Prediction Error", 0.0

def predict_item_and_freshness(image_path):
    """
    Combined prediction: Identify the exact item AND its freshness.
    
    Returns:
        result: dict with item_name, freshness, confidence, confidence_level
    """
    result = {
        "item_name": "Unknown",
        "freshness": "Unknown",
        "confidence": 0.0,
        "confidence_level": "Low"
    }
    
    # Get freshness prediction
    freshness_label, confidence = predict_freshness_with_confidence(image_path)
    
    result["freshness"] = freshness_label
    result["confidence"] = confidence
    result["confidence_level"] = get_confidence_level(confidence)
    
    return result

def get_model_info():
    """Get information about loaded models."""
    info = {
        "freshness_model": "Loaded" if freshness_model else "Not loaded",
        "item_detector": "Loaded" if item_detector else "Not loaded",
        "freshness_path": FRESHNESS_MODEL_PATH,
        "detector_path": ITEM_DETECTOR_PATH
    }
    return info

