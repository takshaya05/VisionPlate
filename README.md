# VISIONPLATE

## Tagline

Smart Plate Recognition

## Overview

VisionPlate is an AI-powered Automatic Number Plate Recognition (ANPR) application that detects vehicle number plates and recognizes their characters from uploaded images. It uses YOLO, EasyOCR, and OpenCV, with advanced low-light enhancement and adaptive recognition to improve plate recognition under challenging lighting conditions.

## Features

- **Plate Detection:** Detect vehicle number plates using YOLO.
- **Plate Recognition:** Recognize number plate characters using EasyOCR and display confidence scores.
- **Night Vision & Low-Light Plate Recovery:** Enhance dark images using OpenCV techniques to improve number plate visibility and recognition.
- **Adaptive Recognition:** Automatically apply suitable image preprocessing based on lighting conditions and recognition confidence.
- **Performance Comparison:** Compare original and enhanced recognition results using accuracy, confidence scores, and processing time.

## Tech Stack

- **Python:** Implements application logic and AI processing.
- **Streamlit:** Builds the interactive web interface and dashboard.
- **YOLO:** Detects vehicle number plates in images.
- **EasyOCR:** Recognizes characters from detected number plates.
- **OpenCV:** Enhances low-light images, improves contrast, and preprocesses images for recognition.

## How to Use

1. **Upload an Image:** Select an image containing one or more vehicles.
2. **Analyze Image Quality:** Evaluate brightness and contrast to identify low-light conditions.
3. **Enhance the Image:** Apply suitable preprocessing to improve number plate visibility.
4. **Detect and Recognize Plates:** Use YOLO to detect number plates and EasyOCR to extract their characters.
5. **Compare Results:** Review recognized plate text, confidence scores, and processing time for the original and enhanced images.