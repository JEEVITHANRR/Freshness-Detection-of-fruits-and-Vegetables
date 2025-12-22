"""
Gemini-based Item Detection and Counting

Uses Google Gemini Vision API for accurate:
- Item identification (exact food names)
- Item counting (accurate counts)
- Categorization (fruit, vegetable, etc.)

This replaces YOLO-World for more accurate results.
"""

import os
import json
import base64
from PIL import Image
import io
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-generativeai not installed. Run: pip install google-generativeai")

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def configure_gemini(api_key=None):
    """Configure Gemini with API key."""
    global GEMINI_API_KEY
    if api_key:
        GEMINI_API_KEY = api_key
    
    if GEMINI_API_KEY and GEMINI_AVAILABLE:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    return False

def image_to_base64(image_path):
    """Convert image to base64 for Gemini."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def detect_and_count_items(image_path):
    """
    Use Gemini Vision to detect and count food items accurately.
    
    Returns:
        dict: {
            "items": [{"name": "apple", "count": 3, "category": "Fruit"}, ...],
            "total_count": 5,
            "raw_response": "..."
        }
    """
    if not GEMINI_AVAILABLE:
        return {"error": "Gemini not available", "items": [], "total_count": 0}
    
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set", "items": [], "total_count": 0}
    
    try:
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Load image
        try:
            img = Image.open(image_path)
        except Exception as e:
            return {"error": f"Failed to load image: {e}", "items": [], "total_count": 0}

        # Try different Gemini models with specific order for free tier quotas
        # 1. Flash 1.5 is most stable with decent quotas
        # 2. Flash 2.0-exp is best but strict quotas
        # 3. Pro Vision is legacy fallback
        # Try different Gemini models available for this key
        # Prioritize Lite/Flash models for better quota/speed
        models_to_try = [
            'gemini-2.5-flash',
            'gemini-2.5-flash-lite',
            'gemini-2.0-flash',
            'gemini-2.0-flash-lite',
            'gemini-2.0-flash-exp',
            'gemini-flash-latest'
        ]
        
        last_error = None
        error_log = []
        
        for model_name in models_to_try:
            try:
                # print(f"🤖 Trying Gemini model: {model_name}...")
                model = genai.GenerativeModel(model_name)
                
                # ... (rest of loop logic) ...
                prompt = """Analyze this image by counting food items AND checking their freshness.

Your task:
1. Identify distinct food items.
2. Count EXACT number of each type.
3. Classify into one of these categories: 
   - Fruit, Vegetable, Bakery, Beverage, Packed Goods, Dairy, Meat, Other.
4. Check EACH item group for spoilage (mold, spots, wrinkles, discoloration). 
   - Be VERY STRICT/PARANOID. 
   - If ANY defect/aging is visible on ANY item in the group, classify the group as "Rotten" or "Slightly Aged".
   - Only classify as "Fresh" if they look PERFECT.

Respond in this EXACT JSON format:
{
    "items": [
        {
            "name": "apple", 
            "count": 3, 
            "category": "Fruit", 
            "freshness": "Fresh", 
            "freshness_confidence": 0.9,
            "observations": "No defects visible"
        },
        {
            "name": "banana", 
            "count": 2, 
            "category": "Fruit", 
            "freshness": "Rotten", 
            "freshness_confidence": 0.95, 
            "observations": "Brown spots and softening visible"
        }
    ],
    "total_count": 5
}

Be PRECISE with counting.
If no food items are visible, return empty items array with total_count 0.
"""
                response = model.generate_content([prompt, img])
                
                # If we get here, it worked
                try:
                    text = response.text
                    if "```json" in text:
                        json_str = text.split("```json")[1].split("```")[0]
                    elif "```" in text:
                        json_str = text.split("```")[1].split("```")[0]
                    else:
                        json_str = text
                    
                    result = json.loads(json_str.strip())
                    result["model_used"] = model_name
                    return result
                except:
                     return {
                        "items": [],
                        "total_count": 0,
                        "error": "Failed to parse API response",
                        "raw": response.text
                    }
            
            except Exception as e:
                # Check for quota error
                error_msg = str(e)
                error_log.append(f"{model_name}: {error_msg}")
                
                if "429" in error_msg or "Quota" in error_msg:
                    time.sleep(5) # Wait longer to respect rate limits
                
                last_error = e
                continue
        
        # All models failed
        return {"items": [], "total_count": 0, "error": f"All models failed. Errors: {'; '.join(error_log)}"}

    except Exception as e:
        return {"items": [], "total_count": 0, "error": str(e)}

def get_item_details(image_path):
    """
    Get detailed information about a single food item.
    
    Returns freshness indicators, ripeness, quality assessment.
    """
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return {"error": "Gemini not configured"}
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try different Gemini models
        model = None
        for model_name in ['gemini-2.0-flash-lite', 'gemini-2.0-flash', 'gemini-2.0-flash-exp']:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except:
                continue
        
        if model is None:
            return {"error": "No compatible Gemini model found"}
        
        img = Image.open(image_path)
        
        prompt = """Analyze this food item image with CRITICAL SCRUTINY for spoilage.

1. **Item Name**: What food item is this?
2. **Freshness Assessment**:
   - LOOK FOR: Mold, dark spots, wrinkling, soft spots, discoloration, bruises.
   - If ANY signs of spoilage are present, classify as "Rotten".
   - If it looks perfect but has minor flaws, classify as "Slightly Aged".
   - Only classify as "Fresh" if it looks PERFECT.
   - Output one of: "Fresh", "Slightly Aged", "Rotten".

3. **Freshness Score**: Rate 1-10 (10 = farm fresh, <5 = inedible)
4. **Visual Signs**: List specific defects found (e.g. "brown spots on skin", "stem mold").
5. **Estimated Shelf Life**: How many days until it should be consumed? (0 if rotten)
6. **Storage Recommendation**: Best way to store this item

Respond in JSON format:
{
    "item_name": "apple",
    "freshness": "Rotten",
    "freshness_score": 3,
    "visual_signs": ["brown spots", "wrinkled skin"],
    "shelf_life_days": 0,
    "storage": "Discard immediately"
}
"""
        
        response = model.generate_content([prompt, img])
        response_text = response.text
        
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text
            
            return json.loads(json_str.strip())
        except:
            return {"raw_response": response_text}
            
    except Exception as e:
        return {"error": str(e)}


# Fallback to YOLO if Gemini not available
def detect_with_fallback(image_path):
    """
    Try Gemini first, fall back to YOLO if unavailable.
    """
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        result = detect_and_count_items(image_path)
        if "error" not in result or result.get("items"):
            return result, "gemini"
    
    # Fallback to YOLO
    try:
        from yolo_detect import detect_objects
        detections, counts, annotated_img = detect_objects(image_path)
        
        items = []
        for name, count in counts.items():
            items.append({
                "name": name,
                "count": count,
                "category": "Fruit" if name in ["apple", "banana", "orange", "grape", "mango"] else "Vegetable"
            })
        
        return {
            "items": items,
            "total_count": sum(counts.values()),
            "detections": detections,
            "annotated_image": annotated_img
        }, "yolo"
        
    except Exception as e:
        return {"error": str(e), "items": [], "total_count": 0}, "error"
