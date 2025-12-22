from ultralytics import YOLO
from collections import Counter
import numpy as np
import math

# Load YOLO-World model (Open Vocabulary Detection)
try:
    model = YOLO("yolov8l-world.pt")  # Large World model
    
    # Define custom vocabulary for Food Freshness
    # This allows detecting items that standard YOLO misses (like Tomato, Potato)
    CUSTOM_CLASSES = [
        "apple", "banana", "orange", "pear", "pineapple", "grape", "strawberry", 
        "watermelon", "lemon", "lime", "peach", "mango", "pomegranate", "kiwi",
        "tomato", "potato", "cucumber", "carrot", "broccoli", "cauliflower", 
        "cabbage", "spinach", "lettuce", "capsicum", "bell pepper", "onion", 
        "garlic", "ginger", "eggplant", "pumpkin", "corn", "mushroom", "radish",
        "zucchini", "sweet potato", "chili"
    ]
    model.set_classes(CUSTOM_CLASSES)
    print(f"✅ YOLO-World loaded with {len(CUSTOM_CLASSES)} custom food classes")
    
except Exception as e:
    print(f"⚠️ YOLO-World failed to load: {e}")
    model = YOLO("yolov8m.pt")

# Category Mapping
CATEGORY_MAP = {
    # Fruits
    "apple": "Fruit", "banana": "Fruit", "orange": "Fruit", "pear": "Fruit", 
    "grape": "Fruit", "strawberry": "Fruit", "watermelon": "Fruit", 
    "lemon": "Fruit", "mango": "Fruit",
    
    # Vegetables
    "broccoli": "Vegetable", "carrot": "Vegetable", "tomato": "Vegetable", 
    "cucumber": "Vegetable", "corn": "Vegetable", "bell pepper": "Vegetable",
    "onion": "Vegetable", "potato": "Vegetable", "eggplant": "Vegetable", 
    "mushroom": "Vegetable", "cauliflower": "Vegetable", "cabbage": "Vegetable",
    "chili": "Vegetable",
}

# STRICT Detection Configuration
DETECTION_CONFIG = {
    "conf": 0.40,              # HIGHER confidence to reduce false positives
    "iou": 0.20,               # LOWER IOU = more aggressive NMS (merges more boxes)
    "min_box_size": 3000,      # Minimum pixel area
    "max_box_ratio": 0.70,     # Max box can be 70% of image (removes full-image detections)
    "center_distance": 50,     # Minimum distance between detection centers (pixels)
    "max_detections": 50,      # Cap detections
}

def get_category(item_name):
    return CATEGORY_MAP.get(item_name.lower(), "Other")

def calculate_box_area(box):
    """Calculate area of a bounding box."""
    return (box[2] - box[0]) * (box[3] - box[1])

def get_box_center(box):
    """Get center point of bounding box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def distance_between_centers(center1, center2):
    """Calculate Euclidean distance between two centers."""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = calculate_box_area(box1)
    area2 = calculate_box_area(box2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def aggressive_nms(detections, iou_threshold=0.3, center_threshold=50):
    """
    Aggressive Non-Maximum Suppression that:
    1. Removes boxes with high IoU overlap
    2. Removes boxes with centers too close together
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (higher first)
    sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    
    keep = []
    for det in sorted_dets:
        should_keep = True
        det_center = get_box_center(det["box"])
        
        for kept in keep:
            # Check IoU overlap
            iou = calculate_iou(det["box"], kept["box"])
            if iou > iou_threshold:
                should_keep = False
                break
            
            # Check center distance (catches cases IoU misses)
            kept_center = get_box_center(kept["box"])
            distance = distance_between_centers(det_center, kept_center)
            if distance < center_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep.append(det)
    
    return keep

def detect_objects(image_path):
    """
    Detect objects with ACCURATE COUNTING using strict NMS and filtering.
    
    Returns:
        detections: List of detected items with details
        counts: Counter of item names
        annotated_image: PIL Image with bounding boxes
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Get image dimensions for ratio filtering
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            img_area = img_width * img_height
    except Exception as e:
        print(f"Error reading image: {e}")
        return [], Counter(), None

    try:
        # Detection with strict settings
        results = model.predict(
            image_path, 
            conf=DETECTION_CONFIG["conf"],
            iou=DETECTION_CONFIG["iou"],
            imgsz=640,  # Standard size for consistency
            augment=False,
            agnostic_nms=True,
            max_det=DETECTION_CONFIG["max_detections"],
            verbose=False
        )
    except Exception as e:
        print(f"Error in YOLO: {e}")
        return [], Counter(), None

    result = results[0]
    boxes = result.boxes
    names = result.names
    
    raw_detections = []
    
    if hasattr(boxes, 'cls') and len(boxes.cls) > 0:
        for i, cls_id in enumerate(boxes.cls.tolist()):
            try:
                name = names[int(cls_id)]
            except:
                continue
            
            # Category filtering - only Fruits and Vegetables
            category = get_category(name)
            if category not in {"Fruit", "Vegetable"}:
                continue

            box = boxes.xyxy[i].tolist()
            conf = boxes.conf[i].item()
            
            # SIZE FILTERING
            box_area = calculate_box_area(box)
            
            # Skip tiny detections
            if box_area < DETECTION_CONFIG["min_box_size"]:
                continue
            
            # Skip if box is too large (likely wrong detection)
            box_ratio = box_area / img_area
            if box_ratio > DETECTION_CONFIG["max_box_ratio"]:
                continue
            
            raw_detections.append({
                "name": name,
                "category": category,
                "box": box,
                "confidence": conf
            })
    
    # AGGRESSIVE NMS - remove duplicates
    detections = aggressive_nms(
        raw_detections, 
        iou_threshold=0.25,  # Very strict
        center_threshold=DETECTION_CONFIG["center_distance"]
    )
    
    # Log filtering results
    filtered_count = len(raw_detections) - len(detections)
    if filtered_count > 0:
        print(f"  NMS removed {filtered_count} duplicate detections")
    
    # Build counts
    item_names = [d["name"] for d in detections]
    counts = Counter(item_names)
    
    # Generate custom annotated image with accurate boxes
    try:
        original = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(original)
        
        # Colors for different items
        COLORS = {
            "apple": "#FF6B6B", "banana": "#FFE66D", "orange": "#FFA94D",
            "tomato": "#FF4757", "carrot": "#FF7F50", "cucumber": "#2ED573",
            "broccoli": "#26DE81", "potato": "#D4A373", "grape": "#A55EEA",
            "mango": "#FFBE76", "lemon": "#F9CA24", "onion": "#DFE6E9",
        }
        
        for det in detections:
            box = det["box"]
            name = det["name"]
            conf = det["confidence"]
            
            color = COLORS.get(name, "#00FF00")
            
            # Draw box
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=3)
            
            # Draw label with background
            label = f"{name} {conf:.0%}"
            text_bbox = draw.textbbox((box[0], box[1] - 20), label)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                          fill=color)
            draw.text((box[0], box[1] - 20), label, fill="white")
        
        # Add total count overlay
        count_text = f"Total: {len(detections)} items"
        draw.rectangle([5, 5, 180, 35], fill="black")
        draw.text((10, 10), count_text, fill="#00FF00")
        
        im_rgb = original
        
    except Exception as e:
        print(f"Error creating annotated image: {e}")
        im_rgb = None
    
    print(f"✅ Detected {len(detections)} items: {dict(counts)}")
    return detections, counts, im_rgb
