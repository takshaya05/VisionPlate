# VISIONPLATE

## Tagline

**Smart Plate Recognition**

## Overview

VisionPlate is an AI-powered Automatic Number Plate Recognition (ANPR) web application that detects vehicle number plates and recognizes their characters automatically.

## Problem Statement

Manual identification of vehicle number plates can be time-consuming and inefficient. VisionPlate uses YOLO for plate detection and EasyOCR for character recognition through an interactive web application.

## Features

* **Plate Detection:** Detect number plates using YOLO.
* **Plate Recognition:** Recognize plate characters using EasyOCR.
* **Confidence Scores:** Display detection and OCR confidence.
* **Multiple Plate Detection:** Detect multiple plates in one image.
* **Interactive Dashboard:** View images, detected plates, and results.

## Tech Stack

* Python
* Streamlit
* YOLO
* EasyOCR
* OpenCV
* NumPy

## Project Structure

* VisionPlate/
  * assets/
  * app.py
  * plate_model.pt
  * requirements.txt
  * README.md