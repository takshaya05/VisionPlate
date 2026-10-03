# VISIONPLATE

## Tagline

**Smart Plate Recognition**

## Overview

VisionPlate is an AI-powered Automatic Number Plate Recognition (ANPR) application that detects vehicle number plates and automatically recognizes their characters from images.

## Features

* **Plate Detection:** Detect vehicle number plates using YOLO.
* **Plate Recognition:** Recognize plate characters using EasyOCR.
* **Confidence Scores:** Display detection and OCR confidence scores.
* **Multiple Plate Detection:** Detect multiple number plates in a single image.
* **Interactive Dashboard:** Display uploaded images, detected plates, and recognition results.

## Tech Stack

* **Python:** Implement the application and AI processing.
* **Streamlit:** Build the interactive web interface.
* **YOLO:** Detect vehicle number plates.
* **EasyOCR:** Recognize characters from detected plates.
* **OpenCV:** Process and manipulate vehicle images.

## Project Structure

* VisionPlate/
  * assets/
  * app.py
  * plate_model.pt
  * requirements.txt