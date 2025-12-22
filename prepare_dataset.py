"""
Dataset Preparation Script for Food Freshness AI

This script:
1. Organizes raw images into proper train/val structure
2. Handles web-downloaded images
3. Creates proper splits for training
4. Validates image integrity
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image

# Configuration
RAW_DATASET = "dataset/web_downloaded"
PROCESSED_DATASET = "dataset/processed"
TRAIN_SPLIT = 0.8  # 80% train, 20% val

# Food categories for item-specific detection training
FOOD_ITEMS = [
    "apple", "banana", "orange", "pear", "pineapple", "grape", "strawberry", 
    "watermelon", "lemon", "lime", "peach", "mango", "pomegranate", "kiwi",
    "tomato", "potato", "cucumber", "carrot", "broccoli", "cauliflower", 
    "cabbage", "spinach", "lettuce", "capsicum", "bell pepper", "onion", 
    "garlic", "ginger", "eggplant", "pumpkin", "corn", "mushroom", "radish",
    "zucchini", "sweet potato", "chili"
]

def validate_image(image_path):
    """Check if an image file is valid."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"  ❌ Invalid image: {image_path} - {e}")
        return False

def create_directory_structure():
    """Create the required directory structure."""
    dirs = [
        f"{PROCESSED_DATASET}/train/fresh",
        f"{PROCESSED_DATASET}/train/rotten",
        f"{PROCESSED_DATASET}/val/fresh",
        f"{PROCESSED_DATASET}/val/rotten",
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  📁 Created: {d}")

def process_web_downloaded():
    """Move web-downloaded images to processed dataset."""
    if not os.path.exists(RAW_DATASET):
        print(f"  ⚠️ Web downloaded folder not found: {RAW_DATASET}")
        return
    
    # Look for fresh/rotten folders in web_downloaded
    categories = ["fresh", "rotten"]
    
    for category in categories:
        source_dir = f"{RAW_DATASET}/train/{category}"
        if not os.path.exists(source_dir):
            continue
            
        images = [f for f in os.listdir(source_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if not images:
            continue
            
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_SPLIT)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to processed
        for img in train_images:
            src = f"{source_dir}/{img}"
            dst = f"{PROCESSED_DATASET}/train/{category}/{img}"
            if validate_image(src):
                shutil.copy2(src, dst)
        
        for img in val_images:
            src = f"{source_dir}/{img}"
            dst = f"{PROCESSED_DATASET}/val/{category}/{img}"
            if validate_image(src):
                shutil.copy2(src, dst)
        
        print(f"  ✅ {category}: {len(train_images)} train, {len(val_images)} val")

def count_dataset():
    """Count images in the processed dataset."""
    print("\n📊 Dataset Summary:")
    print("-" * 40)
    
    total = 0
    for split in ["train", "val"]:
        for category in ["fresh", "rotten"]:
            path = f"{PROCESSED_DATASET}/{split}/{category}"
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                total += count
                print(f"  {split}/{category}: {count} images")
    
    print("-" * 40)
    print(f"  Total: {total} images")
    
    if total < 100:
        print("\n⚠️ Warning: Small dataset. Consider downloading more images.")
        print("   Run: python download_dataset.py")

def remove_duplicates():
    """Remove duplicate images based on file hash."""
    from hashlib import md5
    
    print("\n🔍 Checking for duplicates...")
    
    seen_hashes = {}
    duplicates = []
    
    for split in ["train", "val"]:
        for category in ["fresh", "rotten"]:
            path = f"{PROCESSED_DATASET}/{split}/{category}"
            if not os.path.exists(path):
                continue
                
            for img_file in os.listdir(path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    continue
                    
                img_path = f"{path}/{img_file}"
                with open(img_path, 'rb') as f:
                    file_hash = md5(f.read()).hexdigest()
                
                if file_hash in seen_hashes:
                    duplicates.append(img_path)
                else:
                    seen_hashes[file_hash] = img_path
    
    if duplicates:
        print(f"  Found {len(duplicates)} duplicates. Removing...")
        for dup in duplicates:
            os.remove(dup)
        print(f"  ✅ Removed {len(duplicates)} duplicate images")
    else:
        print("  ✅ No duplicates found")

def main():
    print("=" * 60)
    print("🍎 FOOD FRESHNESS DATASET PREPARATION")
    print("=" * 60)
    
    print("\n1️⃣ Creating directory structure...")
    create_directory_structure()
    
    print("\n2️⃣ Processing web-downloaded images...")
    process_web_downloaded()
    
    print("\n3️⃣ Removing duplicates...")
    remove_duplicates()
    
    print("\n4️⃣ Counting dataset...")
    count_dataset()
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("   Next: Run 'python train_freshness.py' to start training")
    print("=" * 60)

if __name__ == "__main__":
    main()
