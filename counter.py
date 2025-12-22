from ultralytics import YOLO

# Load YOLO model
yolo = YOLO("models/yolov8n.pt")  # Ensure the model path is correct

def count_items(image_path):
    """
    Counts the number of detected objects in an image using YOLOv8.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        int: Number of detected objects.
    """
    results = yolo(image_path, verbose=False)
    
    # Check if boxes exist in the result
    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        return len(results[0].boxes) 
    else:
        return 0

# Example usage
if __name__ == "__main__":
    image_path = "test_image.jpg"
    total_items = count_items(image_path)
    print(f"Total items detected: {total_items}")
