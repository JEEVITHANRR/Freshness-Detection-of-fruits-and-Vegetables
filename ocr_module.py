"""
OCR Module for Product Information Extraction

Extracts:
- Expiry Date
- Batch Number
- Manufacturing Date
- Product Name
- Barcode/QR Code data
"""

import os
import re
from datetime import datetime
from PIL import Image

# Try to import OCR libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ pytesseract not installed. Run: pip install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    reader = None
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️ easyocr not installed. Run: pip install easyocr")

# Date patterns to match
DATE_PATTERNS = [
    r'\d{2}[/-]\d{2}[/-]\d{4}',  # DD/MM/YYYY or DD-MM-YYYY
    r'\d{4}[/-]\d{2}[/-]\d{2}',  # YYYY/MM/DD or YYYY-MM-DD
    r'\d{2}[/-]\d{2}[/-]\d{2}',  # DD/MM/YY or DD-MM-YY
    r'(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s*\d{1,2},?\s*\d{4}',  # Jan 15, 2024
    r'\d{1,2}\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s*\d{4}',  # 15 Jan 2024
]

# Batch number patterns
BATCH_PATTERNS = [
    r'(?:BATCH|LOT|B\.NO|L\.NO)[:\s]*([A-Z0-9]+)',
    r'(?:B|L)[:\s]*([A-Z0-9]{4,})',
]

# Expiry keywords
EXPIRY_KEYWORDS = ['EXP', 'EXPIRY', 'BEST BEFORE', 'USE BY', 'BB', 'EXP DATE']
MFG_KEYWORDS = ['MFG', 'MFD', 'MANUFACTURED', 'PRODUCTION', 'PACKED']


def get_easyocr_reader():
    """Initialize EasyOCR reader."""
    global reader
    if reader is None and EASYOCR_AVAILABLE:
        reader = easyocr.Reader(['en'])
    return reader


def extract_text_tesseract(image_path):
    """Extract text using Tesseract OCR."""
    if not TESSERACT_AVAILABLE:
        return ""
    
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.upper()
    except Exception as e:
        print(f"Tesseract error: {e}")
        return ""


def extract_text_easyocr(image_path):
    """Extract text using EasyOCR."""
    if not EASYOCR_AVAILABLE:
        return ""
    
    try:
        ocr_reader = get_easyocr_reader()
        results = ocr_reader.readtext(image_path)
        text = " ".join([result[1] for result in results])
        return text.upper()
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return ""


def extract_dates(text):
    """Extract all dates from text."""
    dates = []
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    return dates


def extract_batch_number(text):
    """Extract batch/lot number from text."""
    for pattern in BATCH_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return None


def find_expiry_date(text, dates):
    """Find the expiry date from extracted dates."""
    lines = text.split('\n')
    
    for line in lines:
        # Check if line contains expiry keyword
        has_expiry_keyword = any(kw in line for kw in EXPIRY_KEYWORDS)
        
        if has_expiry_keyword:
            # Find date in this line
            for pattern in DATE_PATTERNS:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(0)
    
    # If no explicit expiry found, return the latest date
    if dates:
        return dates[-1]
    
    return None


def find_mfg_date(text, dates):
    """Find the manufacturing date from extracted dates."""
    lines = text.split('\n')
    
    for line in lines:
        has_mfg_keyword = any(kw in line for kw in MFG_KEYWORDS)
        
        if has_mfg_keyword:
            for pattern in DATE_PATTERNS:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(0)
    
    # If no explicit mfg found, return the earliest date
    if dates:
        return dates[0]
    
    return None


def extract_product_info(image_path, use_easyocr=True):
    """
    Extract product information from image.
    
    Args:
        image_path: Path to product image
        use_easyocr: Use EasyOCR (more accurate) or Tesseract
        
    Returns:
        dict: {
            "raw_text": "...",
            "expiry_date": "15/01/2025",
            "mfg_date": "15/01/2024",
            "batch_number": "ABC123",
            "all_dates": [...],
            "ocr_engine": "easyocr"
        }
    """
    # Extract text
    if use_easyocr and EASYOCR_AVAILABLE:
        text = extract_text_easyocr(image_path)
        engine = "easyocr"
    elif TESSERACT_AVAILABLE:
        text = extract_text_tesseract(image_path)
        engine = "tesseract"
    else:
        return {
            "error": "No OCR engine available",
            "raw_text": "",
            "expiry_date": None,
            "mfg_date": None,
            "batch_number": None
        }
    
    # Extract information
    dates = extract_dates(text)
    expiry_date = find_expiry_date(text, dates)
    mfg_date = find_mfg_date(text, dates)
    batch_number = extract_batch_number(text)
    
    return {
        "raw_text": text,
        "expiry_date": expiry_date,
        "mfg_date": mfg_date,
        "batch_number": batch_number,
        "all_dates": dates,
        "ocr_engine": engine
    }


def check_expiry_status(expiry_date_str):
    """
    Check if product is expired.
    
    Returns:
        dict: {
            "is_expired": True/False,
            "days_until_expiry": int (negative if expired),
            "status": "Expired" / "Expiring Soon" / "Fresh"
        }
    """
    if not expiry_date_str:
        return {"status": "Unknown", "is_expired": None, "days_until_expiry": None}
    
    # Try to parse date
    date_formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d",
        "%d/%m/%y", "%d-%m-%y", "%B %d, %Y", "%d %B %Y"
    ]
    
    expiry_date = None
    for fmt in date_formats:
        try:
            expiry_date = datetime.strptime(expiry_date_str, fmt)
            break
        except ValueError:
            continue
    
    if not expiry_date:
        return {"status": "Unknown", "is_expired": None, "days_until_expiry": None}
    
    today = datetime.now()
    days_diff = (expiry_date - today).days
    
    if days_diff < 0:
        status = "Expired"
        is_expired = True
    elif days_diff <= 7:
        status = "Expiring Soon"
        is_expired = False
    else:
        status = "Fresh"
        is_expired = False
    
    return {
        "status": status,
        "is_expired": is_expired,
        "days_until_expiry": days_diff,
        "expiry_date_parsed": expiry_date.strftime("%Y-%m-%d") if expiry_date else None
    }


if __name__ == "__main__":
    # Test OCR
    import sys
    if len(sys.argv) > 1:
        result = extract_product_info(sys.argv[1])
        print(f"OCR Result: {result}")
