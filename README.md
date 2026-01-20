# 🛣️ Road Damage Severity Detection System
📌 Project Overview

This project focuses on detecting and classifying road damage severity using computer vision techniques.
By leveraging a deep learning–based object detection model, the system identifies road defects such as potholes and cracks and categorizes them into different severity levels to assist in infrastructure monitoring and maintenance planning.
The solution supports real-time image inference through an interactive web application.

🎯 Problem Statement
Manual road inspection is time-consuming, expensive, and prone to human error. Undetected or delayed road damage can lead to accidents, increased vehicle wear, and higher maintenance costs.
Objective:
Develop an automated system to detect road damage and assess its severity from images to enable faster and more efficient road condition monitoring.

🧠 Solution Approach
1. Data Preparation
Used 260+ labeled road damage images for training
Validated model performance on 2000+ real-world road images
Performed image preprocessing and normalization

2. Model Development
Fine-tuned a YOLOv8 object detection model
Optimized confidence thresholds to reduce false positives
Leveraged PyTorch-based YOLO architecture for efficient inference

4. Severity Classification
Classified detected damage into three severity levels (Low, Medium, High)
Severity determined based on:
Detected damage area
Model confidence score

4. Deployment
Built an interactive Streamlit web application
Enabled real-time image upload and damage detection
Achieved low-latency inference suitable for practical use

🛠️ Tech Stack
Language: Python
Deep Learning: YOLOv8 (PyTorch)
Libraries: NumPy, PIL
Web Framework: Streamlit

📊 Key Results
Accurate detection of road damages in diverse lighting and road conditions
Reliable severity classification across three levels
Low-latency inference enabling real-time user interaction
Robust performance on unseen real-world images

🖥️ Application Feature
Upload road images for instant analysis
Bounding box visualization of detected damage
Severity level display for each detected region

Dataset - https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection
