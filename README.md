# Crop Disease Detection System

An end-to-end Machine Learning and Web Application designed to automatically identify and diagnose crop diseases from plant leaf images. This project aims to help farmers detect plant anomalies early, minimizing crop yield loss and optimizing agricultural productivity.

---

## Project Overview

This repository contains a full-stack deployment architecture for deep learning-based crop disease diagnosis:
* **Deep Learning Models:** Trained Convolutional Neural Networks (CNNs) optimized for precise leaf classification.
* **Backend API:** A high-performance FastAPI/Flask server handling image processing and model inference.
* **Frontend Web App:** An intuitive, responsive user interface built for easy image uploads and instant results layout.

---

## Repository Structure

```text
crop_disease_detection_2/
├── api/                   # Backend server, API routing, and controller logic
├── frontend/              # Client-side web application interface
├── models/                # Saved trained model files (.h5, .keras, or saved_model format)
├── training/              # Jupyter notebooks, datasets, and model training pipelines
├── anaconda_projects/     # Environment configurations and project dependencies
└── Untitled.ipynb         # Experimental/scratchpad analysis notebook
```

---

##  Technology Stack

* **Machine Learning:** Python, TensorFlow / Keras, OpenCV, NumPy, Pandas
* **Backend API:** FastAPI (or Flask), Uvicorn
* **Frontend:** React.js / HTML5 / CSS3 / JavaScript (Tailwind CSS)
* **Environment:** Anaconda / Conda Virtual Environments

---

## Installation & Setup

Follow these steps to spin up the project locally:

### 1. Prerequisites
Ensure you have [Anaconda](https://anaconda.com) and Python 3.8+ installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com
cd crop_disease_detection_2
```

### 3. Backend Setup (API)
```bash
cd api
# Install the required packages
pip install -r requirements.txt
# Start the backend server
uvicorn main:app --reload
```

### 4. Frontend Setup
```bash
cd ../frontend
# Install node packages
npm install
# Launch the local development server
npm start
```

---

## Model & Training Details

* **Dataset:** Utilises standard agricultural plant village imagery datasets.
* **Classes Detected:** Covers standard crop variations (e.g., Tomato Bacterial Spot, Potato Early Blight, Corn Common Rust).
* **Architecture:** Custom CNN / Transfer Learning via MobileNet/ResNet models for edge-device optimization.

---

## Contributing

Contributions are welcome! If you want to improve the model accuracy, expand dataset coverage, or upgrade the UI:
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request
