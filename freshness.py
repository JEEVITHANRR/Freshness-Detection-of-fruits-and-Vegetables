from datetime import datetime, timedelta
import random

# Base Shelf Life (in days) for Fresh items
SHELF_LIFE_RULES = {
    "apple": 14, "banana": 5, "orange": 10, "grape": 7,
    "tomato": 5, "potato": 21, "onion": 30, "carrot": 14,
    "bread": 4, "milk": 7, "egg": 21, "default": 7
}

def estimate_shelf_life(product_name, freshness_label, confidence):
    """
    Estimate shelf life based on product type and freshness status.
    Args:
        product_name (str): Name of the product (e.g., 'apple')
        freshness_label (str): 'Fresh', 'Rotten', etc.
        confidence (float): Confidence score of the freshness model (0-100)
    
    Returns:
        dict: {
            "status": "Fresh" | "Medium" | "Near Spoilage" | "Spoiled",
            "days_left": int,
            "expiry_date": str (YYYY-MM-DD)
        }
    """
    base_days = SHELF_LIFE_RULES.get(product_name.lower(), SHELF_LIFE_RULES["default"])
    
    # Determine Status
    if "Rotten" in freshness_label:
        status = "Spoiled"
        days_left = 0
    else:
        # If Fresh, fine-tune based on confidence (simulating visual defects)
        # If confidence is low (e.g., < 70%), maybe it's "Medium" fresh
        if confidence > 85:
            status = "Fresh"
            days_left = base_days
        elif confidence > 60:
            status = "Medium"
            days_left = int(base_days * 0.6)
        else:
            status = "Near Spoilage"
            days_left = int(base_days * 0.2) + 1

    expiry_date = (datetime.now() + timedelta(days=days_left)).strftime("%Y-%m-%d")

    return {
        "status": status,
        "days_left": days_left,
        "expiry_date": expiry_date
    }
