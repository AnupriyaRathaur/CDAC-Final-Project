🚗 Automatic Number Plate Recognition (ANPR)
📌 Project Overview

Automatic Number Plate Recognition (ANPR) is a computer vision–based system that automatically detects vehicle number plates from images and videos and extracts the alphanumeric characters present on the plates.

This project implements an end-to-end ANPR system using Machine Learning and Deep Learning techniques. The system supports image-based and video-based detection, works on CPU as well as GPU, and provides an interactive web interface using Streamlit.

The ANPR system is designed to be scalable, efficient, and suitable for real-world traffic surveillance applications.

🎯 Objectives

Detect vehicle number plates from images and videos

Recognize alphanumeric characters from detected plates

Support both CPU and GPU execution

Provide real-time video processing capability

Deploy an interactive web-based interface using Streamlit

🧠 System Architecture

The ANPR system follows a modular architecture:

Image / Video Input
        ↓
Preprocessing
        ↓
License Plate Detection (YOLO)
        ↓
Plate Cropping
        ↓
Text Recognition (OCR)
        ↓
Final Output


Each module works independently, making the system easy to maintain and extend.

🛠️ Technology Stack
🔹 Programming Language

Python

🔹 Machine Learning / Deep Learning

YOLO (You Only Look Once) for license plate detection

PyTorch as the deep learning backend

🔹 OCR (Text Recognition)

EasyOCR for extracting alphanumeric text from plates

🔹 Computer Vision

OpenCV for image and video processing

🔹 Frontend

Streamlit for interactive web-based UI

📂 Dataset Description

Dataset Name: Car Plate Detection Dataset

Source: Kaggle

Data Type: Vehicle images with annotated license plates

License: Open (CC0)

The dataset contains real-world vehicle images along with bounding box annotations for license plates. The data is split into training and validation sets to evaluate model performance.

🧹 Data Preprocessing

The following preprocessing steps are applied:

Removal of corrupt and invalid images

Validation of annotations

Conversion of annotation format for model compatibility

Splitting data into training and validation sets

Preprocessing ensures clean and consistent data for effective model training.

🤖 Model Description

The system uses a YOLO-based object detection model to detect license plates.

Why YOLO?

Real-time detection capability

High accuracy

Single-pass object detection

Suitable for video processing

YOLO identifies the location of the number plate, while OCR identifies the text on the plate.

🔤 Optical Character Recognition (OCR)

OCR is used to extract the alphanumeric characters from detected license plates.

Why OCR is Needed?

Detection alone gives only the plate location

OCR converts visual text into machine-readable text

EasyOCR is chosen because it:

Supports deep learning–based recognition

Works on both CPU and GPU

Handles different fonts and styles

⚡ CPU vs GPU Execution

The system supports both CPU and GPU execution.

Feature	CPU	GPU
Training Speed	Slow	Fast
Inference Speed	Moderate	High
Hardware Cost	Low	High
Real-time Video	Limited	Efficient

GPU acceleration significantly improves performance for large datasets and real-time video processing.

🧪 Testing

The system is tested on:

Static Images: To verify detection accuracy

Videos: To demonstrate real-time ANPR capability

Each video frame is processed independently for detection and recognition.

🌐 Streamlit Web Application

The project includes a Streamlit-based web application that allows:

Uploading images for number plate detection

Uploading videos for real-time processing

Displaying detected plates with recognized text

This makes the project suitable for live demos and presentations.

🚀 Deployment

The Streamlit application can be deployed using:

Streamlit Community Cloud

Local network deployment

Temporary public deployment using tunneling tools

Deployment provides a public URL to access the ANPR system.

https://cdac-ml-project-parallelcomputing.streamlit.app/

📌 Applications

Traffic monitoring and surveillance

Toll booth automation

Parking management systems

Law enforcement

Smart city infrastructure

🔮 Future Enhancements

Improve OCR accuracy for low-resolution images

Support multi-camera inputs

Integrate database for vehicle record storage

Cloud and edge deployment

Night-time and low-light detection improvements

📚 References

Kaggle Vehicle License Plate Dataset

YOLO Object Detection Documentation

OpenCV Documentation

EasyOCR Documentation

Research papers on ANPR and computer vision

👩‍💻 Author

Name: Anupriya Rathaur
Project Type: CDAC Training Project
Domain: Machine Learning & Computer Vision

🙏 Acknowledgement

I would like to thank my project guide for their support and guidance in completing this project successfully.
