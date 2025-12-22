"""
Result Fusion & Shelf-Life Estimation Module

Combines results from:
- Gemini/YOLO detection
- ResNet18 freshness
- OCR product info

To produce unified estimates of shelf life.
"""

from datetime import datetime, timedelta

# Base shelf life data (in days) for different items when fresh
SHELF_LIFE_DATA = {
    # Fruits
    "apple": {"fresh": 14, "refrigerated": 30, "frozen": 240},
    "banana": {"fresh": 5, "refrigerated": 7, "frozen": 180},
    "orange": {"fresh": 10, "refrigerated": 21, "frozen": 300},
    "grape": {"fresh": 3, "refrigerated": 7, "frozen": 180},
    "strawberry": {"fresh": 2, "refrigerated": 5, "frozen": 180},
    "mango": {"fresh": 5, "refrigerated": 7, "frozen": 180},
    "pear": {"fresh": 5, "refrigerated": 14, "frozen": 180},
    "lemon": {"fresh": 21, "refrigerated": 45, "frozen": 120},
    "watermelon": {"fresh": 7, "refrigerated": 14, "frozen": 180},
    
    # Vegetables
    "tomato": {"fresh": 5, "refrigerated": 10, "frozen": 180},
    "potato": {"fresh": 21, "refrigerated": 60, "frozen": 270},
    "carrot": {"fresh": 7, "refrigerated": 21, "frozen": 360},
    "cucumber": {"fresh": 3, "refrigerated": 7, "frozen": 180},
    "broccoli": {"fresh": 2, "refrigerated": 5, "frozen": 270},
    "cauliflower": {"fresh": 3, "refrigerated": 7, "frozen": 270},
    "onion": {"fresh": 30, "refrigerated": 60, "frozen": 180},
    "cabbage": {"fresh": 7, "refrigerated": 14, "frozen": 180},
    "spinach": {"fresh": 2, "refrigerated": 5, "frozen": 180},
    "bell pepper": {"fresh": 5, "refrigerated": 10, "frozen": 180},
    "eggplant": {"fresh": 3, "refrigerated": 7, "frozen": 180},
    "mushroom": {"fresh": 2, "refrigerated": 7, "frozen": 180},
    "corn": {"fresh": 3, "refrigerated": 7, "frozen": 270},
    
    # Bakery
    "bread": {"fresh": 5, "refrigerated": 14, "frozen": 90},
    "croissant": {"fresh": 2, "refrigerated": 5, "frozen": 30},
    "cake": {"fresh": 3, "refrigerated": 7, "frozen": 90},
    "cookie": {"fresh": 21, "refrigerated": 60, "frozen": 180},
    "bagel": {"fresh": 3, "refrigerated": 7, "frozen": 90},
    
    # Dairy
    "milk": {"fresh": 7, "refrigerated": 7, "frozen": 30},
    "cheese": {"fresh": 14, "refrigerated": 30, "frozen": 180},
    "yogurt": {"fresh": 7, "refrigerated": 14, "frozen": 60},
    "butter": {"fresh": 30, "refrigerated": 90, "frozen": 270},
    
    # Beverages
    "juice": {"fresh": 10, "refrigerated": 14, "frozen": 180},
    "soda": {"fresh": 180, "refrigerated": 180, "frozen": 180},
    "water": {"fresh": 365, "refrigerated": 365, "frozen": 365},
    
    # Packed Goods
    "canned": {"fresh": 365, "refrigerated": 365, "frozen": 365},
    "jar": {"fresh": 180, "refrigerated": 365, "frozen": 365},
    "packet": {"fresh": 90, "refrigerated": 180, "frozen": 180},
    "snack": {"fresh": 60, "refrigerated": 90, "frozen": 180},
    "chocolate": {"fresh": 180, "refrigerated": 365, "frozen": 365},

    # Default
    "default": {"fresh": 5, "refrigerated": 10, "frozen": 180}
}

# Freshness decay factors
FRESHNESS_MODIFIERS = {
    "Fresh": 1.0,        # Full shelf life
    "Slightly Aged": 0.5,  # Half shelf life remaining
    "Rotten": 0.0,       # No shelf life
    "Unknown": 0.7       # Conservative estimate
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 0.85,
    "medium": 0.60,
    "low": 0.40
}


def get_base_shelf_life(item_name, storage="fresh"):
    """Get base shelf life for an item."""
    item_key = item_name.lower()
    if item_key in SHELF_LIFE_DATA:
        return SHELF_LIFE_DATA[item_key].get(storage, SHELF_LIFE_DATA["default"][storage])
    return SHELF_LIFE_DATA["default"].get(storage, 5)


def calculate_shelf_life(item_name, freshness_label, freshness_confidence, 
                         ocr_expiry_date=None, storage="refrigerated"):
    """
    Calculate estimated shelf life combining all available data.
    
    Args:
        item_name: Name of the food item
        freshness_label: "Fresh", "Rotten", "Slightly Aged", or "Unknown"
        freshness_confidence: Confidence of freshness prediction (0-1)
        ocr_expiry_date: Expiry date from OCR (optional)
        storage: Storage method ("fresh", "refrigerated", "frozen")
    
    Returns:
        dict: {
            "estimated_days": int,
            "expiry_date": "YYYY-MM-DD",
            "confidence": "high"/"medium"/"low",
            "recommendation": str,
            "source": "ocr"/"ai"/"combined"
        }
    """
    today = datetime.now()
    
    # If OCR provides expiry date, use it as primary source
    if ocr_expiry_date:
        try:
            # Parse OCR date
            date_formats = [
                "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d",
                "%d/%m/%y", "%d-%m-%y"
            ]
            exp_date = None
            for fmt in date_formats:
                try:
                    exp_date = datetime.strptime(ocr_expiry_date, fmt)
                    break
                except:
                    continue
            
            if exp_date:
                days_remaining = (exp_date - today).days
                
                # Apply freshness modifier (if item looks bad, reduce even if date says OK)
                if freshness_label == "Rotten" and freshness_confidence > 0.7:
                    days_remaining = min(days_remaining, 0)  # Override to expired
                    recommendation = "Do not consume - item appears spoiled despite label date"
                elif freshness_label == "Slightly Aged" and freshness_confidence > 0.6:
                    days_remaining = min(days_remaining, int(days_remaining * 0.5))
                    recommendation = "Consume soon - item showing signs of aging"
                else:
                    recommendation = get_recommendation(days_remaining)
                
                return {
                    "estimated_days": days_remaining,
                    "expiry_date": exp_date.strftime("%Y-%m-%d"),
                    "confidence": "high",
                    "recommendation": recommendation,
                    "source": "combined" if freshness_label != "Unknown" else "ocr"
                }
        except Exception as e:
            pass  # Fall through to AI-based estimation
    
    # AI-based estimation
    base_days = get_base_shelf_life(item_name, storage)
    modifier = FRESHNESS_MODIFIERS.get(freshness_label, 0.7)
    
    estimated_days = int(base_days * modifier)
    expiry_date = (today + timedelta(days=estimated_days)).strftime("%Y-%m-%d")
    
    # Determine confidence level
    if freshness_confidence >= CONFIDENCE_THRESHOLDS["high"]:
        confidence = "high"
    elif freshness_confidence >= CONFIDENCE_THRESHOLDS["medium"]:
        confidence = "medium"
    else:
        confidence = "low"
    
    recommendation = get_recommendation(estimated_days, freshness_label)
    
    return {
        "estimated_days": estimated_days,
        "expiry_date": expiry_date,
        "confidence": confidence,
        "recommendation": recommendation,
        "source": "ai"
    }


def get_recommendation(days_remaining, freshness_label=None):
    """Get storage/consumption recommendation."""
    if days_remaining <= 0:
        return "⚠️ Do not consume - item has expired or is spoiled"
    elif days_remaining <= 1:
        return "🔴 Consume today or discard"
    elif days_remaining <= 3:
        return "🟠 Consume within 1-3 days"
    elif days_remaining <= 7:
        return "🟡 Best consumed within a week"
    else:
        return "🟢 Item is fresh - proper storage recommended"


def fuse_results(detection_result, freshness_result, ocr_result=None):
    """
    Fuse all analysis results into a unified output.
    
    Args:
        detection_result: From gemini_detect or yolo_detect
        freshness_result: From resnet_freshness
        ocr_result: From ocr_module (optional)
    
    Returns:
        dict: Unified analysis result
    """
    items_with_analysis = []
    
    # Get detected items
    items = detection_result.get("items", [])
    total_count = detection_result.get("total_count", 0)
    
    for item in items:
        item_name = item.get("name", "unknown")
        count = item.get("count", 1)
        category = item.get("category", "Food")
        
        # Get freshness info
        # Prefer specific freshness from Gemini detection if available
        if item.get("freshness"):
            freshness_label = item.get("freshness")
            # Default to high confidence if Gemini said so but didn't give score
            freshness_conf = item.get("freshness_confidence", 0.9) 
        else:
            # Fallback to ResNet global result
            freshness_label = freshness_result.get("label", "Unknown")
            freshness_conf = freshness_result.get("confidence", 0.5)
        
        # Get OCR expiry if available
        ocr_expiry = None
        if ocr_result:
            ocr_expiry = ocr_result.get("expiry_date")
        
        # Calculate shelf life
        shelf_life = calculate_shelf_life(
            item_name, 
            freshness_label, 
            freshness_conf,
            ocr_expiry
        )
        
        items_with_analysis.append({
            "name": item_name,
            "count": count,
            "category": category,
            "freshness": {
                "label": freshness_label,
                "confidence": freshness_conf,
                "display": f"{freshness_label} ({freshness_conf*100:.0f}%)"
            },
            "shelf_life": shelf_life,
            "ocr_info": {
                "expiry_date": ocr_result.get("expiry_date") if ocr_result else None,
                "batch_number": ocr_result.get("batch_number") if ocr_result else None
            },
            "observations": item.get("observations") or item.get("notes") # Pass through gemini notes
        })
    
    return {
        "total_items": total_count,
        "items": items_with_analysis,
        "summary": generate_summary(items_with_analysis),
        "timestamp": datetime.now().isoformat()
    }


def generate_summary(items):
    """Generate a summary of the analysis."""
    if not items:
        return "No food items detected"
    
    fresh_count = sum(1 for i in items if "Fresh" in i["freshness"]["label"])
    rotten_count = sum(1 for i in items if "Rotten" in i["freshness"]["label"])
    expiring_soon = sum(1 for i in items if i["shelf_life"]["estimated_days"] <= 3)
    
    summary_parts = []
    
    if fresh_count > 0:
        summary_parts.append(f"✅ {fresh_count} fresh item(s)")
    if rotten_count > 0:
        summary_parts.append(f"❌ {rotten_count} rotten item(s)")
    if expiring_soon > 0:
        summary_parts.append(f"⚠️ {expiring_soon} expiring soon")
    
    return " | ".join(summary_parts) if summary_parts else "Analysis complete"
