Urban Grid Watch

**AI-Powered Traffic Monitoring & Congestion Prediction System**

Urban Grid Watch is a complete AI-driven system that detects vehicles from traffic videos, classifies congestion levels, and predicts future traffic trends using machine learning.
This project was built as part of my **HCLTech Summer Internship (2025)**.

---

## Features

✔️ Upload any CCTV/traffic video
✔️ Extract frames using OpenCV
✔️ Detect vehicles using **YOLOv8 (COCO + Custom Roboflow model)**
✔️ Calculate **weighted congestion index**
✔️ Store data in **MySQL**
✔️ Classify congestion levels using **KMeans (5 clusters)**
✔️ Train a **Random Forest Classifier**
✔️ Predict **next 5-hour congestion levels**
✔️ Display everything in a **web dashboard**
✔️ Show confidence scores and performance metrics

---

## System Pipeline

```
Video Upload → Frame Extraction → YOLOv8 Detection → MySQL Storage 
     → Weighted Count Calculation → KMeans Clustering 
     → Random Forest Prediction → 5-Hour Forecasting 
     → Dashboard Visualization
```

---

## Project Structure (Short Overview)

* `test2.py` → Frame extraction + YOLO detection
* `phase2_clustering.py` → KMeans clustering + labeling
* `phase3_prediction.py` → Model training with GridSearch
* `congestion_predictor.py` → Future 5-hour forecasting
* `templates/` → HTML dashboard
* `static/` → CSS, JS, processed videos
* MySQL database with 7 tables (videos, frames, vehicle_counts, predictions, etc.)

---


## Dashboard Features

Built using **HTML, CSS, JavaScript, Chart.js**.

Shows:

* Hour-wise congestion predictions
* Confidence graph
* YOLO detected video
* ML performance metrics
* Vehicle count tables

---

## Tech Stack

### Backend & ML

* Python
* YOLOv8
* OpenCV
* Scikit-learn
* Numpy, Pandas
* Joblib
* Flask

### Frontend

* HTML
* CSS
* JavaScript
* Chart.js

### Database

* MySQL

---

## How to Run

### 1️ Install Dependencies

```
pip install -r requirements.txt
```

### 2️ Configure MySQL

Update database settings in code.

### 3️ Run YOLO Detection

```
python test2.py
```

### 4️ Run Clustering

```
python phase2_clustering.py
```

### 5️ Train ML Model

```
python phase3_prediction.py
```

### 6️ Generate 5-Hour Forecast

```
python congestion_predictor.py
```

### 7️ Launch Dashboard

```
python app.py
```

---

