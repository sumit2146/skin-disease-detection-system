# **Skin Disease Detection System**

A comprehensive Flask-based web application that leverages Deep Learning to identify various skin conditions from user-uploaded images. This system is designed with role-based access for Patients, Doctors, and Admins, facilitating streamlined diagnostics, medical reporting, and administrative management.

## **🚀 Key Features**

* **AI-Driven Diagnostics**: Integrates a TensorFlow/Keras CNN model (.h5) to analyze lesion images and provide instant predictions.  
* **Role-Based Access Control (RBAC)**:  
  * **Patient**: Upload skin images for analysis, view history, and download PDF reports.  
  * **Doctor**: Review diagnostic records across all patients for clinical verification.  
  * **Admin**: Full system oversight, including user management, analytics tracking, and on-the-fly model updates via the web interface.  
* **Automated Medical Reports**: Generates professional PDF reports using ReportLab, detailing the diagnosis, confidence levels, and curated medical/dietary advice.  
* **Feedback Management**: Integrated channel for users to provide feedback directly to the administration team.  
* **Robust Inference Engine**: Includes a "Mock Mode" that ensures the application remains functional for demonstration purposes even if specialized ML environments (TensorFlow) are unavailable.

## **📂 Project Architecture**

├── app/  
│   ├── static/          \# CSS, system images, and user-uploaded scans  
│   ├── templates/       \# Jinja2 HTML templates  
│   ├── \_\_init\_\_.py      \# Flask app factory and SQLAlchemy initialization  
│   ├── auth.py          \# User authentication and session management  
│   ├── main.py          \# Core application routing and business logic  
│   └── models.py        \# Database schema (Users, Scans, Feedback)  
├── ml/  
│   ├── inference.py     \# Image preprocessing and model prediction logic  
│   ├── mapping.py       \# Knowledge base for disease info and treatments  
│   └── train.py         \# Utilities for model training  
├── class\_indices.json   \# Mapping of model output neurons to disease names  
├── requirements.txt     \# List of Python library dependencies  
├── run.py               \# Main entry point to launch the server  
└── skin\_disease\_model.h5 \# The trained neural network model file

## **🛠️ Installation & Setup**

1. **Clone the Project**:  
   git clone \<your-repository-link\>  
   cd skin-disease-detection-system

2. Install Required Dependencies:  
   It is recommended to use a virtual environment.  
   pip install \-r requirements.txt

3. Initialize the Database:  
   The application uses SQLite for simplicity. The database file (db.sqlite) is automatically generated upon the first execution of the app.

## **🏃 Execution**

1. **Launch the Flask Server**:  
   python run.py

2. **Access the Dashboard**: Open your browser and navigate to http://127.0.0.1:5000.  
3. **Getting Started**: Register as a new user. The system initializes with standard roles; administrative roles can be assigned via the database or the admin panel for existing administrators.

## **🧪 Machine Learning Technicals**

* **Model Format**: Keras H5 (skin\_disease\_model.h5).  
* **Preprocessing**: Images are resized to $224 \\times 224$ pixels and normalized according to MobileNetV2 standards.  
* **Knowledge Base**: Disease information, including dietary advice ("Eat" and "Avoid") and common treatments, is managed in ml/mapping.py to ensure users receive actionable insights alongside their results.

## **⚠️ Medical Disclaimer**

**Important**: This software is intended for educational and research purposes only. The automated detections provided by the AI model are not clinical diagnoses. Users should always consult with a certified medical professional or dermatologist for accurate diagnosis and treatment plans.