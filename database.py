"""
SQLite Database Module

Handles:
- User management
- Inventory tracking
- Scan history
- Analytics
"""

import sqlite3
import os
from datetime import datetime
import json

DATABASE_PATH = "food_freshness.db"

def get_connection():
    """Get database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT,
            email TEXT,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Inventory table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            item_name TEXT NOT NULL,
            category TEXT,
            quantity INTEGER DEFAULT 1,
            freshness TEXT,
            freshness_score REAL,
            expiry_date TEXT,
            batch_number TEXT,
            image_path TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Scan history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            scan_type TEXT,
            items_detected TEXT,
            total_count INTEGER,
            freshness_results TEXT,
            ocr_results TEXT,
            detection_method TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            inventory_id INTEGER,
            alert_type TEXT,
            message TEXT,
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (inventory_id) REFERENCES inventory(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

# User operations
def create_user(username, password_hash, name="", email="", role="user"):
    """Create a new user."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (username, password_hash, name, email, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, password_hash, name, email, role))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_user(username):
    """Get user by username."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def update_last_login(user_id):
    """Update user's last login timestamp."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users SET last_login = ? WHERE id = ?
    ''', (datetime.now(), user_id))
    conn.commit()
    conn.close()

# Inventory operations
def add_inventory_item(user_id, item_name, category, quantity=1, freshness=None, 
                       freshness_score=None, expiry_date=None, batch_number=None, image_path=None):
    """Add item to inventory."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO inventory (user_id, item_name, category, quantity, freshness, 
                              freshness_score, expiry_date, batch_number, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, item_name, category, quantity, freshness, freshness_score, 
          expiry_date, batch_number, image_path))
    conn.commit()
    item_id = cursor.lastrowid
    conn.close()
    return item_id

def get_user_inventory(user_id, status='active'):
    """Get user's inventory items."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM inventory WHERE user_id = ? AND status = ?
        ORDER BY added_at DESC
    ''', (user_id, status))
    items = cursor.fetchall()
    conn.close()
    return [dict(item) for item in items]

def update_inventory_item(item_id, **kwargs):
    """Update inventory item."""
    if not kwargs:
        return False
    
    conn = get_connection()
    cursor = conn.cursor()
    
    set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
    values = list(kwargs.values()) + [item_id]
    
    cursor.execute(f'''
        UPDATE inventory SET {set_clause}, updated_at = CURRENT_TIMESTAMP 
        WHERE id = ?
    ''', values)
    conn.commit()
    conn.close()
    return True

def delete_inventory_item(item_id):
    """Soft delete inventory item."""
    return update_inventory_item(item_id, status='deleted')

# Scan history operations
def save_scan(user_id, image_path, scan_type, items_detected, total_count,
              freshness_results=None, ocr_results=None, detection_method="yolo"):
    """Save a scan result."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO scan_history (user_id, image_path, scan_type, items_detected,
                                  total_count, freshness_results, ocr_results, detection_method)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, image_path, scan_type, json.dumps(items_detected), total_count,
          json.dumps(freshness_results) if freshness_results else None,
          json.dumps(ocr_results) if ocr_results else None, detection_method))
    conn.commit()
    scan_id = cursor.lastrowid
    conn.close()
    return scan_id

def get_user_scan_history(user_id, limit=50):
    """Get user's scan history."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM scan_history WHERE user_id = ?
        ORDER BY created_at DESC LIMIT ?
    ''', (user_id, limit))
    scans = cursor.fetchall()
    conn.close()
    
    result = []
    for scan in scans:
        scan_dict = dict(scan)
        scan_dict['items_detected'] = json.loads(scan_dict['items_detected']) if scan_dict['items_detected'] else []
        scan_dict['freshness_results'] = json.loads(scan_dict['freshness_results']) if scan_dict['freshness_results'] else None
        scan_dict['ocr_results'] = json.loads(scan_dict['ocr_results']) if scan_dict['ocr_results'] else None
        result.append(scan_dict)
    
    return result

# Alert operations
def create_alert(user_id, inventory_id, alert_type, message):
    """Create an alert."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO alerts (user_id, inventory_id, alert_type, message)
        VALUES (?, ?, ?, ?)
    ''', (user_id, inventory_id, alert_type, message))
    conn.commit()
    conn.close()

def get_user_alerts(user_id, unread_only=False):
    """Get user's alerts."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if unread_only:
        cursor.execute('''
            SELECT * FROM alerts WHERE user_id = ? AND is_read = 0
            ORDER BY created_at DESC
        ''', (user_id,))
    else:
        cursor.execute('''
            SELECT * FROM alerts WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
    
    alerts = cursor.fetchall()
    conn.close()
    return [dict(alert) for alert in alerts]

def mark_alert_read(alert_id):
    """Mark alert as read."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE alerts SET is_read = 1 WHERE id = ?', (alert_id,))
    conn.commit()
    conn.close()

# Analytics
def get_user_stats(user_id):
    """Get user statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total scans
    cursor.execute('SELECT COUNT(*) as count FROM scan_history WHERE user_id = ?', (user_id,))
    total_scans = cursor.fetchone()['count']
    
    # Total items detected
    cursor.execute('SELECT SUM(total_count) as total FROM scan_history WHERE user_id = ?', (user_id,))
    total_items = cursor.fetchone()['total'] or 0
    
    # Inventory count
    cursor.execute("SELECT COUNT(*) as count FROM inventory WHERE user_id = ? AND status = 'active'", (user_id,))
    inventory_count = cursor.fetchone()['count']
    
    # Freshness breakdown
    cursor.execute('''
        SELECT freshness, COUNT(*) as count 
        FROM inventory WHERE user_id = ? AND status = 'active'
        GROUP BY freshness
    ''', (user_id,))
    freshness_breakdown = {row['freshness']: row['count'] for row in cursor.fetchall()}
    
    # Items by category
    cursor.execute('''
        SELECT category, COUNT(*) as count 
        FROM inventory WHERE user_id = ? AND status = 'active'
        GROUP BY category
    ''', (user_id,))
    category_breakdown = {row['category']: row['count'] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        "total_scans": total_scans,
        "total_items_detected": total_items,
        "inventory_count": inventory_count,
        "freshness_breakdown": freshness_breakdown,
        "category_breakdown": category_breakdown
    }

def get_expiring_items(user_id, days=7):
    """Get items expiring within specified days."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM inventory 
        WHERE user_id = ? AND status = 'active' AND expiry_date IS NOT NULL
        ORDER BY expiry_date ASC
    ''', (user_id,))
    
    items = cursor.fetchall()
    conn.close()
    
    # Filter by expiry date (within days)
    expiring = []
    today = datetime.now()
    
    for item in items:
        item_dict = dict(item)
        try:
            exp_date = datetime.strptime(item_dict['expiry_date'], "%Y-%m-%d")
            days_until = (exp_date - today).days
            if days_until <= days:
                item_dict['days_until_expiry'] = days_until
                expiring.append(item_dict)
        except:
            pass
    
    return expiring

# Initialize database on import
init_database()
