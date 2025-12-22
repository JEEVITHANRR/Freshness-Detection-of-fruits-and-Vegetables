from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime, timedelta
from PIL import Image
from collections import Counter

# Custom Modules
from model import predict_freshness_with_confidence
from yolo_detect import detect_objects
from freshness import estimate_shelf_life
from storage import save_prediction, get_user_history, get_all_stats, verify_user, create_user

app = Flask(__name__)
app.secret_key = "freshness-ai-secret-key"

# Configuration
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- ROUTES -----------------

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = verify_user(username, password)
        if user:
            session["user_id"] = user["username"]
            session["role"] = user["role"]
            session["name"] = user["name"]
            session["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials", "danger")
            
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("fullname")
        username = request.form.get("username")
        password = request.form.get("password")
        
        if create_user(username, password, name):
            flash("Account created! Please login.", "success")
            return redirect(url_for("login"))
        else:
            flash("Username already exists.", "danger")
            
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    stats = get_all_stats()
    user_history = get_user_history(session["user_id"])
    
    return render_template("dashboard.html", 
                           user=session, 
                           stats=stats, 
                           recent_uploads=user_history[-5:]) # Show last 5

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    user_history = get_user_history(session["user_id"])
    # Reverse to show newest first
    return render_template("history.html", history=user_history[::-1])

@app.route("/settings")
def settings():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("settings.html", user=session)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    if request.method == "POST":
        file = request.files.get("image")
        if not file or not allowed_file(file.filename):
            flash("Invalid or missing file", "danger")
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        
        # --- AI PIPELINE NEW ---
        try:
            # Import new modules here to avoid circular dependencies if any
            from gemini_detect import detect_with_fallback
            from resnet_freshness import predict_freshness
            from ocr_module import extract_product_info
            from result_fusion import fuse_results
            
            # 1. Detect Objects (Gemini or YOLO)
            detection_result, method = detect_with_fallback(path)
            
            # 2. Freshness Analysis
            # If Gemini provided freshness, we can skip ResNet
            freshness_result = {}
            has_gemini_freshness = any(i.get('freshness') for i in detection_result.get('items', []))
            
            if not has_gemini_freshness:
                freshness_result = predict_freshness(path)
            
            # 3. OCR Analysis
            ocr_result = extract_product_info(path)
            
            # 4. Fuse Results
            fusion = fuse_results(detection_result, freshness_result, ocr_result)
            
            # Format for template
            processed_data = []
            for item in fusion['items']:
                processed_data.append({
                    "name": item['name'],
                    "category": item['category'],
                    "confidence": item['freshness']['confidence'] * 100, # Convert to %
                    "freshness": item['freshness']['label'],
                    "shelf_life": item['shelf_life']['estimated_days'],
                    "count": item['count'],
                    "observations": item.get('observations')
                })
            
            total_items = fusion['total_items']
            if detection_result.get('annotated_image'):
                annotated_filename = f"annotated_{filename}"
                annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
                detection_result['annotated_image'].save(annotated_path)
                display_image = annotated_filename
            else:
                display_image = filename
                
        except Exception as e:
            print(f"New Pipeline Error: {e}")
            flash(f"Error in analysis: {e}", "danger")
            processed_data = []
            total_items = 0
            display_image = filename

        # 5. Save results to DB
        from storage import save_prediction 
        for item in processed_data:
            save_prediction(
                session["user_id"], filename, item["name"], item["category"],
                item["freshness"], "Active", item["confidence"], 
                (datetime.now() + timedelta(days=item["shelf_life"])).strftime("%Y-%m-%d")
            )

        # Calculate Stats for display
        stats = {
            "total_items": total_items,
            "fresh_count": sum(1 for i in processed_data if "Fresh" in i["freshness"]),
            "rotten_count": sum(1 for i in processed_data if "Rotten" in i["freshness"]),
            "expiring_soon": sum(1 for i in processed_data if i["shelf_life"] <= 3)
        }
        
        # Calculate summary counts
        freshness_stats = Counter()
        item_freshness_breakdown = {}
        
        for data in processed_data:
            f_label = data["freshness"]
            freshness_stats[f_label] += data["count"] # Use count from fusion
            
            name = data["name"]
            if name not in item_freshness_breakdown:
                item_freshness_breakdown[name] = Counter()
        flash(f"Processed {len(detections)} items!", "success")
        return render_template("result.html", 
                               image=display_image, 
                               details=processed_data, 
                               counts=counts,
                               freshness_stats=freshness_stats,
                               item_freshness_breakdown=item_freshness_breakdown)

    return render_template("upload.html")

@app.route("/api/stats")
def api_stats():
    # Return JSON for charts
    stats = get_all_stats() # This is summary, we might need granular data for charts.
    # For now returning summary.
    return jsonify(stats)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(debug=True, port=port)
