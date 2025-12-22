"""
Streamlit Visual Interface for Food Freshness AI

Features:
- Dashboard with analytics
- Image upload for scanning
- Camera capture
- Inventory management
- Alerts and notifications
- History view
"""

import streamlit as st
import os
from datetime import datetime
from PIL import Image
import io

# Import modules
try:
    from gemini_detect import detect_and_count_items, get_item_details, configure_gemini
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

try:
    from resnet_freshness import predict_freshness
    RESNET_AVAILABLE = True
except:
    RESNET_AVAILABLE = False

try:
    from ocr_module import extract_product_info, check_expiry_status
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

try:
    from yolo_detect import detect_objects
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

import database as db

# Page config
st.set_page_config(
    page_title="Food Freshness AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .fresh-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
    }
    .rotten-badge {
        background-color: #F44336;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None

# Sidebar navigation
def sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/apple.png", width=80)
        st.title("🍎 Food Freshness AI")
        
        if st.session_state.user_id:
            st.success(f"Welcome, {st.session_state.username}!")
            
            page = st.radio(
                "Navigation",
                ["📊 Dashboard", "📷 Scan Food", "📦 Inventory", "📜 History", "⚙️ Settings"]
            )
            
            if st.button("🚪 Logout"):
                st.session_state.user_id = None
                st.session_state.username = None
                st.rerun()
            
            return page
        else:
            return "login"

# Login page
def login_page():
    st.markdown("<h1 class='main-header'>🍎 Food Freshness AI</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Login", use_container_width=True):
                    user = db.get_user(username)
                    if user and user['password_hash'] == password:  # Simple check, use proper hashing in production
                        st.session_state.user_id = user['id']
                        st.session_state.username = user['username']
                        db.update_last_login(user['id'])
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username")
                new_name = st.text_input("Full Name")
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password", key="signup_pwd")
                
                if st.form_submit_button("Create Account", use_container_width=True):
                    if new_username and new_password:
                        user_id = db.create_user(new_username, new_password, new_name, new_email)
                        if user_id:
                            st.success("Account created! Please login.")
                        else:
                            st.error("Username already exists")
                    else:
                        st.warning("Please fill in required fields")

# Dashboard
def dashboard_page():
    st.markdown("<h1 class='main-header'>📊 Dashboard</h1>", unsafe_allow_html=True)
    
    # Get stats
    stats = db.get_user_stats(st.session_state.user_id)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Scans", stats['total_scans'], delta=None)
    
    with col2:
        st.metric("Items Detected", stats['total_items_detected'])
    
    with col3:
        st.metric("Inventory Items", stats['inventory_count'])
    
    with col4:
        fresh_count = stats['freshness_breakdown'].get('Fresh', 0)
        rotten_count = stats['freshness_breakdown'].get('Rotten', 0)
        st.metric("Fresh Items", fresh_count, delta=f"-{rotten_count} rotten" if rotten_count > 0 else None)
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Freshness Distribution")
        if stats['freshness_breakdown']:
            import pandas as pd
            df = pd.DataFrame.from_dict(stats['freshness_breakdown'], orient='index', columns=['Count'])
            st.bar_chart(df)
        else:
            st.info("No data yet. Start scanning!")
    
    with col2:
        st.subheader("📊 Items by Category")
        if stats['category_breakdown']:
            import pandas as pd
            df = pd.DataFrame.from_dict(stats['category_breakdown'], orient='index', columns=['Count'])
            st.bar_chart(df)
        else:
            st.info("No data yet. Start scanning!")
    
    # Alerts
    st.divider()
    st.subheader("⚠️ Expiring Soon")
    
    expiring = db.get_expiring_items(st.session_state.user_id, days=7)
    if expiring:
        for item in expiring[:5]:
            days = item.get('days_until_expiry', 0)
            if days < 0:
                st.error(f"🚨 **{item['item_name']}** expired {abs(days)} days ago!")
            elif days == 0:
                st.warning(f"⚠️ **{item['item_name']}** expires TODAY!")
            else:
                st.warning(f"📅 **{item['item_name']}** expires in {days} days")
    else:
        st.success("✅ No items expiring soon")

# Scan page
def scan_page():
    st.markdown("<h1 class='main-header'>📷 Scan Food Items</h1>", unsafe_allow_html=True)
    
    # Detection method selection
    col1, col2 = st.columns(2)
    
    with col1:
        detection_method = st.selectbox(
            "Detection Method",
            ["Gemini AI (Recommended)", "YOLO Detection"],
            index=0 if GEMINI_AVAILABLE else 1
        )
    
    with col2:
        scan_type = st.selectbox(
            "Scan Type",
            ["Count Items", "Check Freshness", "Read Labels (OCR)", "Full Scan"]
        )
    
    # Gemini API Key input
    if "Gemini" in detection_method:
        api_key = st.text_input("Gemini API Key", type="password", 
                                help="Get your API key from https://makersuite.google.com/app/apikey")
        if api_key:
            configure_gemini(api_key)
    
    st.divider()
    
    # Image input options
    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Camera"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload food image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            # Save uploaded file
            upload_dir = "static/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Uploaded Image")
                st.image(uploaded_file)
            
            with col2:
                st.subheader("🔍 Analysis Results")
                
                with st.spinner("Analyzing..."):
                    # Run detection based on method
                    if "Gemini" in detection_method and GEMINI_AVAILABLE:
                        result = detect_and_count_items(file_path)
                        method = "gemini"
                    else:
                        detections, counts, annotated = detect_objects(file_path)
                        result = {
                            "items": [{"name": name, "count": count, "category": "Food"} 
                                     for name, count in counts.items()],
                            "total_count": sum(counts.values())
                        }
                        method = "yolo"
                    
                    # Display results
                    if result.get("error"):
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success(f"✅ Detected **{result['total_count']}** items")
                        
                        # Show items
                        if result.get("items"):
                            for item in result["items"]:
                                item_info = f"• **{item['name']}**: {item['count']} ({item.get('category', 'Food')})"
                                
                                # Display Gemin freshness if available
                                if item.get('freshness'):
                                    freshness = item['freshness']
                                    conf_str = f"({item.get('freshness_confidence', 0.9)*100:.0f}%)" if item.get('freshness_confidence') else ""
                                    if "Rotten" in freshness:
                                        item_info += f" - <span class='rotten-badge'>{freshness} {conf_str}</span>"
                                    else:
                                        item_info += f" - <span class='fresh-badge'>{freshness} {conf_str}</span>"
                                    
                                    if item.get('observations'):
                                         item_info += f" <br>&nbsp;&nbsp;<i>📝 {item['observations']}</i>"
                                
                                st.markdown(item_info, unsafe_allow_html=True)
                        
                        # Freshness check (Legacy / ResNet fallback)
                        # Only run if Gemini didn't provide freshness
                        items_have_freshness = any(i.get('freshness') for i in result.get("items", []))
                        
                        if (scan_type in ["Check Freshness", "Full Scan"] and 
                            RESNET_AVAILABLE and 
                            not items_have_freshness):
                            
                            st.divider()
                            st.subheader("🍃 Freshness Analysis (ResNet)")
                            
                            freshness_result = predict_freshness(file_path)
                            label = freshness_result['label']
                            conf = freshness_result['confidence'] * 100
                            
                            if label == "Fresh":
                                st.success(f"✅ **{label}** ({conf:.1f}% confident)")
                            else:
                                st.error(f"❌ **{label}** ({conf:.1f}% confident)")
                        
                        # OCR
                        if scan_type in ["Read Labels (OCR)", "Full Scan"] and OCR_AVAILABLE:
                            st.divider()
                            st.subheader("📝 Product Information")
                            
                            ocr_result = extract_product_info(file_path)
                            
                            if ocr_result.get('expiry_date'):
                                st.write(f"**Expiry Date:** {ocr_result['expiry_date']}")
                                
                                expiry_status = check_expiry_status(ocr_result['expiry_date'])
                                if expiry_status['is_expired']:
                                    st.error(f"⚠️ {expiry_status['status']}")
                                else:
                                    st.success(f"✅ {expiry_status['status']} ({expiry_status['days_until_expiry']} days left)")
                            
                            if ocr_result.get('batch_number'):
                                st.write(f"**Batch Number:** {ocr_result['batch_number']}")
                        
                        # Save to database
                        st.divider()
                        if st.button("💾 Save to Inventory", use_container_width=True):
                            # Save scan history
                            db.save_scan(
                                st.session_state.user_id,
                                file_path,
                                scan_type,
                                result.get("items", []),
                                result.get("total_count", 0),
                                detection_method=method
                            )
                            
                            # Add items to inventory
                            for item in result.get("items", []):
                                db.add_inventory_item(
                                    st.session_state.user_id,
                                    item['name'],
                                    item.get('category', 'Food'),
                                    item.get('count', 1),
                                    image_path=file_path
                                )
                            
                            st.success("✅ Saved to inventory!")
    
    with tab2:
        camera_image = st.camera_input("Take a picture")
        
        if camera_image:
            # Save camera image
            upload_dir = "static/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            with open(file_path, "wb") as f:
                f.write(camera_image.getbuffer())
            
            st.info(f"Image saved. Click 'Analyze' to process.")
            
            if st.button("🔍 Analyze", use_container_width=True):
                # Similar processing as upload tab
                st.info("Processing... (same as upload flow)")

# Inventory page
def inventory_page():
    st.markdown("<h1 class='main-header'>📦 Inventory</h1>", unsafe_allow_html=True)
    
    # Get inventory
    inventory = db.get_user_inventory(st.session_state.user_id)
    
    if not inventory:
        st.info("Your inventory is empty. Start scanning food items!")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category_filter = st.selectbox("Category", ["All"] + list(set(i['category'] or 'Other' for i in inventory)))
    
    with col2:
        freshness_filter = st.selectbox("Freshness", ["All", "Fresh", "Rotten", "Unknown"])
    
    with col3:
        sort_by = st.selectbox("Sort By", ["Date Added", "Name", "Expiry Date"])
    
    # Display inventory
    st.divider()
    
    for item in inventory:
        if category_filter != "All" and item['category'] != category_filter:
            continue
        if freshness_filter != "All" and item['freshness'] != freshness_filter:
            continue
        
        with st.expander(f"🍎 {item['item_name']} ({item['quantity']})", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Category:** {item['category']}")
                st.write(f"**Quantity:** {item['quantity']}")
            
            with col2:
                freshness = item['freshness'] or 'Unknown'
                if freshness == 'Fresh':
                    st.success(f"**Freshness:** {freshness}")
                elif freshness == 'Rotten':
                    st.error(f"**Freshness:** {freshness}")
                else:
                    st.info(f"**Freshness:** {freshness}")
            
            with col3:
                if item['expiry_date']:
                    st.write(f"**Expiry:** {item['expiry_date']}")
                
                if st.button("🗑️ Remove", key=f"del_{item['id']}"):
                    db.delete_inventory_item(item['id'])
                    st.rerun()

# History page
def history_page():
    st.markdown("<h1 class='main-header'>📜 Scan History</h1>", unsafe_allow_html=True)
    
    history = db.get_user_scan_history(st.session_state.user_id)
    
    if not history:
        st.info("No scan history yet. Start scanning!")
        return
    
    for scan in history:
        with st.expander(f"📷 Scan on {scan['created_at']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type:** {scan['scan_type']}")
                st.write(f"**Method:** {scan['detection_method']}")
                st.write(f"**Items Found:** {scan['total_count']}")
            
            with col2:
                if scan['items_detected']:
                    st.write("**Detected Items:**")
                    for item in scan['items_detected']:
                        st.write(f"• {item.get('name', 'Unknown')}: {item.get('count', 1)}")

# Settings page
def settings_page():
    st.markdown("<h1 class='main-header'>⚙️ Settings</h1>", unsafe_allow_html=True)
    
    st.subheader("🔑 API Configuration")
    
    gemini_key = st.text_input("Gemini API Key", type="password")
    if gemini_key:
        st.success("API Key configured")
        configure_gemini(gemini_key)
    
    st.divider()
    
    st.subheader("🔔 Notifications")
    
    st.checkbox("Enable expiry alerts", value=True)
    st.slider("Alert me before expiry (days)", 1, 14, 3)
    
    st.divider()
    
    st.subheader("🎨 Appearance")
    
    st.selectbox("Theme", ["Light", "Dark", "Auto"])
    
    st.divider()
    
    st.subheader("🗄️ Data Management")
    
    if st.button("📤 Export Data"):
        st.info("Export feature coming soon!")
    
    if st.button("🗑️ Clear History", type="secondary"):
        st.warning("This will delete all your scan history. Are you sure?")

# Main app
def main():
    page = sidebar()
    
    if page == "login" or not st.session_state.user_id:
        login_page()
    elif page == "📊 Dashboard":
        dashboard_page()
    elif page == "📷 Scan Food":
        scan_page()
    elif page == "📦 Inventory":
        inventory_page()
    elif page == "📜 History":
        history_page()
    elif page == "⚙️ Settings":
        settings_page()

if __name__ == "__main__":
    main()
