import pandas as pd
import os
from datetime import datetime

DATA_FILE = "data.xlsx"
USERS_FILE = "users.xlsx"

def init_user_db():
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame(columns=["username", "password", "name", "role", "created_at"])
        # Add default admin
        df.loc[0] = ["admin", "admin123", "System Admin", "Admin", datetime.now().strftime("%Y-%m-%d")]
        df.to_excel(USERS_FILE, index=False)

def verify_user(username, password):
    if not os.path.exists(USERS_FILE):
        init_user_db()
    
    df = pd.read_excel(USERS_FILE)
    # Ensure columns exist (handling legacy/empty files)
    if "username" not in df.columns: return None

    user = df[df["username"].astype(str) == str(username)]
    if not user.empty and str(user.iloc[0]["password"]) == str(password):
        return user.iloc[0].to_dict()
    return None

def create_user(username, password, name, role="Staff"):
    if not os.path.exists(USERS_FILE):
        init_user_db()
    
    df = pd.read_excel(USERS_FILE)
    if not df[df["username"].astype(str) == str(username)].empty:
        return False # User exists

    new_user = {
        "username": username,
        "password": password,
        "name": name,
        "role": role,
        "created_at": datetime.now().strftime("%Y-%m-%d")
    }
    df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
    df.to_excel(USERS_FILE, index=False)
    return True

def init_db():
    """Initialize the Excel database if it doesn't exist."""
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=[
            "UploadID", "Timestamp", "UserID", "ImageFile", 
            "Product", "Category", "Freshness", "ShelfLife", 
            "Confidence", "ExpiryDate"
        ])
        df.to_excel(DATA_FILE, index=False)

def save_prediction(user_id, image_file, product, category, freshness, shelf_life, confidence, expiry_date):
    """Append a new prediction record to the Excel file."""
    if not os.path.exists(DATA_FILE):
        init_db()

    new_data = {
        "UploadID": [datetime.now().strftime("%Y%m%d%H%M%S")],
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "UserID": [user_id],
        "ImageFile": [image_file],
        "Product": [product],
        "Category": [category],
        "Freshness": [freshness],
        "ShelfLife": [shelf_life],
        "Confidence": [confidence],
        "ExpiryDate": [expiry_date]
    }

    df_new = pd.DataFrame(new_data)
    
    # Check if file exists and append
    try:
        df_existing = pd.read_excel(DATA_FILE)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    except Exception:
        df_final = df_new

    df_final.to_excel(DATA_FILE, index=False)

def get_user_history(user_id):
    """Retrieve history for a specific user."""
    if not os.path.exists(DATA_FILE):
        return []
    
    df = pd.read_excel(DATA_FILE)
    if "UserID" not in df.columns:
        return []

    user_data = df[df["UserID"] == user_id]
    user_data = user_data.fillna("") # Replace NaNs with empty strings to prevent Jinja errors
    return user_data.to_dict(orient="records")

def get_all_stats():
    """Retrieve aggregate statistics for the dashboard."""
    if not os.path.exists(DATA_FILE):
        return {"total": 0, "fresh": 0, "expired": 0, "near_expiry": 0, "categories": {}}
    
    df = pd.read_excel(DATA_FILE)
    
    stats = {
        "total": len(df),
        "fresh": len(df[df["Freshness"].str.contains("Fresh", case=False, na=False)]),
        "expired": len(df[df["Freshness"].str.contains("Spoiled|Rotten", case=False, na=False)]),
        "near_expiry": len(df[df["Freshness"].str.contains("Medium", case=False, na=False)]),
        "categories": df["Category"].value_counts().to_dict()
    }
    return stats
