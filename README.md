<img width="1381" height="755" alt="Screenshot 2026-04-25 at 3 50 26 PM" src="https://github.com/user-attachments/assets/c039178a-1e60-41cf-a221-94cc10c98485" /><img width="1388" height="753" alt="Screenshot 2026-04-25 at 3 55 38 PM" src="https://github.com/user-attachments/assets/48a1ec81-adc4-47ec-9ab4-9102435b67f4" />
<img width="1386" height="758" alt="Screenshot 2026-04-25 at 3 53 48 PM" src="https://github.com/user-attachments/assets/00522511-37f8-49d5-b29a-3480b594bb09" />
<img width="1387" height="757" alt="Screenshot 2026-04-25 at 3 53 32 PM" src="https://github.com/user-attachments/assets/eed894c9-6d25-40b7-b896-bfe4daf1bdb7" />
<img width="1385" height="756" alt="Screenshot 2026-04-25 at 3 53 06 PM" src="https://github.com/user-attachments/assets/44eb5c50-f1a0-4d3d-b0e5-5488ce0d750c" />
<img width="1386" height="753" alt="Screenshot 2026-04-25 at 3 52 27 PM" src="https://github.com/user-attachments/assets/5588097d-787b-45f1-b165-352a77220573" />
<img width="1384" height="750" alt="Screenshot 2026-04-25 at 3 51 57 PM" src="https://github.com/user-attachments/assets/ddb5236a-1daf-4e13-9f0a-f09010914b43" />
<img width="1386" height="756" alt="Screenshot 2026-04-25 at 3 51 09 PM" src="https://github.com/user-attachments/assets/c1ba8216-bf05-4d00-8bc6-8a8f0483a572" />
<img width="1387" height="755" alt="Screenshot 2026-04-25 at 3 50 48 PM" src="https://github.com/user-attachments/assets/6b513edc-66fd-4752-8c37-ca0fe9f2f912" />


<h1 align="center">
  🍎 Food Freshness AI
</h1>

<p align="center">
  <strong>AI-Powered Food Quality Detection & Inventory Management System</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Flask-2.0+-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Gemini_AI-Powered-4285F4?logo=google&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/YOLOv8-Detection-00FFFF?logo=yolo&logoColor=black" alt="YOLO">
  <img src="https://img.shields.io/badge/ResNet18-Classification-FF6F00" alt="ResNet">
</p>

---

## 📋 Overview

**Food Freshness AI** is an intelligent system that uses multiple AI models to detect, classify, and assess the freshness of food items. Perfect for grocery stores, restaurants, food warehouses, and home use to reduce food waste and ensure quality.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Smart Detection** | Gemini Vision AI for accurate food identification and counting |
| 🍃 **Freshness Analysis** | ResNet18 deep learning model for Fresh/Rotten classification |
| 📦 **Object Detection** | YOLOv8 for real-time food item detection |
| 📝 **OCR Scanning** | Extract expiry dates and product information from labels |
| 📊 **Analytics Dashboard** | Visual charts and statistics for inventory insights |
| 🗄️ **Inventory Management** | Track, manage, and monitor food items |
| ⚠️ **Expiry Alerts** | Automatic notifications for items expiring soon |
| 📷 **Camera Support** | Capture images directly from webcam |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                         │
│  ┌─────────────────┐       ┌─────────────────────────────┐ │
│  │   Flask Web App │       │      Streamlit Dashboard    │ │
│  └────────┬────────┘       └──────────────┬──────────────┘ │
└───────────┼────────────────────────────────┼───────────────┘
            │                                │
┌───────────▼────────────────────────────────▼───────────────┐
│                    AI Detection Layer                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐│
│  │ Gemini AI  │  │  YOLOv8    │  │   ResNet18 Freshness   ││
│  │ (Counting) │  │ (Objects)  │  │   (Classification)     ││
│  └────────────┘  └────────────┘  └────────────────────────┘│
│  ┌────────────────────────────────────────────────────────┐│
│  │              OCR Module (EasyOCR/Tesseract)            ││
│  └────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────────┐
│                     Data Layer                               │
│  ┌────────────────┐           ┌────────────────────────────┐│
│  │   SQLite DB    │           │     File Storage           ││
│  │   (Inventory)  │           │  (Images, Models, Results) ││
│  └────────────────┘           └────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Webcam (optional, for camera capture)
- [Gemini API Key](https://aistudio.google.com/app/apikey) (for AI detection)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/food-freshness-ai.git
   cd food-freshness-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

5. **Run the application**
   
   **Option A: Streamlit (Recommended)**
   ```bash
   streamlit run streamlit_app.py
   ```
   
   **Option B: Flask**
   ```bash
   python app.py
   ```

6. **Open in browser**
   - Streamlit: http://localhost:8501
   - Flask: http://localhost:5003

---

## 📁 Project Structure

```
food-freshness-ai/
│
├── 📱 Applications
│   ├── streamlit_app.py      # Streamlit dashboard (main UI)
│   └── app.py                # Flask web application
│
├── 🤖 AI Modules
│   ├── gemini_detect.py      # Gemini Vision for detection & counting
│   ├── resnet_freshness.py   # ResNet18 freshness classifier
│   ├── yolo_detect.py        # YOLOv8 object detection
│   ├── ocr_module.py         # OCR for label reading
│   └── result_fusion.py      # Combine results from multiple models
│
├── 🎓 Training
│   ├── train_freshness.py    # Train freshness classifier
│   ├── train_item_detector.py # Train custom YOLO model
│   ├── prepare_dataset.py    # Dataset preparation utilities
│   └── download_dataset.py   # Download training images
│
├── 💾 Data
│   ├── database.py           # SQLite database operations
│   ├── storage.py            # File storage utilities
│   ├── dataset/              # Training dataset
│   └── models/               # Trained model weights
│
├── 📊 Visualization
│   ├── visualize_results.py  # Generate ML charts
│   └── results/              # Analysis results & charts
│
├── 🌐 Web
│   ├── templates/            # HTML templates (Flask)
│   └── static/               # CSS, JS, uploads
│
└── ⚙️ Config
    ├── requirements.txt      # Python dependencies
    ├── .env.example          # Environment template
    └── .env                  # Your configuration
```

---

## 🎯 Features in Detail

### 1. 📷 Food Scanning

Upload or capture images to detect and analyze food items:

- **Item Detection**: Identifies what food items are in the image
- **Counting**: Accurately counts each type of item
- **Categorization**: Groups items (Fruits, Vegetables, Dairy, etc.)
- **Freshness Assessment**: Determines if items are Fresh, Slightly Aged, or Rotten

### 2. 🍃 Freshness Classification

Deep learning model trained to detect spoilage:

| Status | Description |
|--------|-------------|
| 🟢 **Fresh** | No visible defects, safe to consume |
| 🟡 **Slightly Aged** | Minor imperfections, consume soon |
| 🔴 **Rotten** | Visible spoilage, should be discarded |

### 3. 📊 Analytics Dashboard

Visual insights into your inventory:

- **Freshness Distribution** - Pie chart of fresh vs rotten items
- **Category Breakdown** - Bar chart by food category
- **Top Items** - Most frequently scanned items
- **Scan Activity** - Timeline of scanning activity

### 4. 📝 OCR Label Reading

Extract information from product labels:

- Expiry dates
- Batch numbers
- Manufacturing dates
- Product names

### 5. ⚠️ Smart Alerts

Get notified about:

- Items expiring today
- Items expiring within 7 days
- Already expired items

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
# Required: Get from https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your-gemini-api-key

# Optional: Flask settings
FLASK_PORT=5003
FLASK_DEBUG=true
```

### Detection Methods

| Method | Pros | Cons |
|--------|------|------|
| **Gemini AI** | Most accurate, detailed analysis | Requires API key, rate limits |
| **YOLOv8** | Fast, works offline | Less accurate for counting |
| **ResNet18** | Accurate freshness detection | Single item focus |

---

## 🎓 Training Custom Models

### Train Freshness Classifier

1. Prepare your dataset:
   ```
   dataset/
   ├── train/
   │   ├── fresh/
   │   └── rotten/
   └── val/
       ├── fresh/
       └── rotten/
   ```

2. Run training:
   ```bash
   python train_freshness.py
   ```

3. The trained model will be saved to `models/resnet18_freshness.pth`

### Download More Training Data

```bash
python download_dataset.py
```

This will scrape images from the web for various food categories.

---

## 📸 Screenshots

### Dashboard
The main dashboard shows key metrics and charts for your inventory.

### Scan Page
Upload or capture images to analyze food items.

### Inventory
View and manage all tracked food items.

### Analytics
Detailed charts and reports for data analysis.

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Flask + Jinja2, HTML/CSS |
| **Backend** | Python, SQLite |
| **AI/ML** | PyTorch, YOLOv8, ResNet18, Google Gemini |
| **OCR** | EasyOCR, Tesseract |
| **Data** | NumPy, Pandas, OpenCV |
| **Visualization** | Matplotlib, Seaborn |

---

## 📊 Model Performance

### Freshness Classifier (ResNet18)

| Metric | Score |
|--------|-------|
| Accuracy | ~92% |
| Precision | ~91% |
| Recall | ~93% |
| F1-Score | ~92% |

*Performance varies based on training data and food types.*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev/) for Vision AI capabilities
- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [PyTorch](https://pytorch.org/) for deep learning framework
- [Streamlit](https://streamlit.io/) for the amazing dashboard framework

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

<p align="center">
  Made with ❤️ for reducing food waste
</p>

<p align="center">
  ⭐ Star this repo if you find it helpful!
</p>
