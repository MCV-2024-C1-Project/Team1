# Team 1 - MCV - C1 project

## Table of Contents
1. [Context](#context)
2. [Artwork Recognition Project](#artwork-recognition-project)
3. [Project Overview](#project-overview)
   - [Final Objective](#final-objective)
   - [Week 2 Goal](#week-2-goal)
4. [Key Components](#key-components)
5. [Requirements](#requirements)
6. [Repository Structure](#repository-structure)
   - [Files](#files)
   - [Notebook](#notebook)

## Context

With the development of digital image processing technology, it has become imperative to find methods to efficiently search and browse images from large image collections. Generally, three categories of methods for image retrieval are used: **text-based**, **content-based**, and **semantic-based**. 

In daily life, people search for images mainly via search engines such as Google, Yahoo, etc., which are primarily based on text keyword searches. Prompted by market demand for search services, image retrieval has become an extremely active research area in the fields of pattern recognition and artificial intelligence. 

Current image retrieval techniques are usually based on low-level features (e.g., color, texture, shape, spatial layout). However, low-level features often fail to describe high-level semantic concepts; thus, a 'semantic gap' exists between high-level concepts and low-level features. To reduce this 'semantic gap', researchers have adopted machine-learning techniques to derive high-level semantics.

**However**, in this task, we will focus on low-level techniques, specifically those involving histograms.

## Artwork Recognition Project

This project, developed over four weeks by **Team 1**, aims to create a model that can recognize a painting from a museum's database based on a given image. Upon recognition, the system will return the name of the painting and the artist. 

This project is being developed by **Team 1** using ![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

## Project Overview

The project is structured into four phases divided in weeks, each contributing to the final goal of building an efficient artwork recognition model.

### Final Objective
- **Goal**: Given an image of a painting from a museum, the model will identify the corresponding artwork in the database and return the painting's name and its author.

## Week 3 Goal

- **Goal**: Given a museum dataset and two different query sets, we have to carry out several tasks.
- **Tasks**: The system will utlize the following methods:
  1. **Noise filtering**: with QSD1-W3 (images with noise or color changes), the denoising class detects noise based on image gradients from the database and applies adaptive denoising methods (Gaussian, bilateral, or Wiener filtering) with performance metrics (PSNR, SSIM) to enhance image quality.
  2. **Texture descriptors**: with QSD1_W3 (images with noise or color changes), the feature extractor class extracts texture descriptors from images using the following techniques: block-based histograms, Discrete Cosine Transform (DCT), Local Binary Pattern (LBP) and wavelet transforms across multiple color spaces and channels.
  3. **Background removal**: with QSD2_W3 (images with a background) our goal was to use the GrabCut algorithm, supported by morphological operations, to  segment foreground objects (paintings) from complex backgrounds, providing cleaner binary masks for accurate contour detection and cropping of paintings.

### Key Components:
1. **Image Preprocessing**:
- Denoising: remove noise from images, possibly using wavelet-based techniques to retain essential features.
- Color Space Transformation: transform images into different color spaces (e.g., HSV, YCrCb, Gray) to enhance feature extraction and similarity calculations.
2. **Feature Extraction**:
- Wavelet Descriptors: extracting features through wavelet decomposition in order to capture the texture and color distribution of the images.
- DCT: coefficients are computed for each image and are then used as discrete descriptors.
- LBP: capture the local structure by comparing each pixel to its neighbors.
3. **Segmentation**:
- GrabCut Algorithm: Used to separate foreground (paintings) from the background, leveraging Gaussian Mixture Models (GMMs) and graph cuts.
- Morphological Operations: Dilation and erosion steps to refine binary masks by filling gaps, removing noise, and enhancing key contours.

## Requirements

To run this project, make sure you have the following libraries installed:

![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)
![Scipy](https://img.shields.io/badge/scipy-FF6633?style=for-the-badge&logo=spicy&logoColor=white)
![Pickle](https://img.shields.io/badge/Pickle-0a9c6b?style=for-the-badge&logo=python&logoColor=white)
![PyWavelets](https://img.shields.io/badge/PyWavelets-1d8bcd?style=for-the-badge&logo=python&logoColor=white)
![SCIKIT-IMAGE](https://img.shields.io/badge/scikit--image-5b80b1?style=for-the-badge&logo=python&logoColor=white)


You can install the required libraries with the following command:

```bash
pip install numpy opencv-python plotly scipy
```
OR 

Use the requirements.txt file 
```bash
pip install -r requirements.txt
```
Note: Make sure with the second method you have python version >=3.10

## Repository Structure

This repository contains nine Python files, each serving a specific role in the project.

### Files:

1. **data_loader.py**
   - Responsible for loading images from the dataset and preparing them for feature extraction.

2. **evaluation.py**
   - Contains the evaluation metrics and functions to assess the performance of the retrieval system.

3. **features_extractor.py**
   - Handles the extraction of features from images, such as color histograms, across various color spaces.

4. **retrieval_system.py**
   - Implements the main retrieval system that, given a query image, retrieves the most similar images from the dataset.

5. **similarity_calculator.py**
   - Computes similarity scores between histograms using different metrics to rank the results. In this file, we have implemented both OpenCV and SciPy because we found different metrics that can be useful throughout the project.
  
6. **denoising.py**
   - Responsible for denoising the images.
  
7. **evaluation_funcs.py**
   - Evaluates the system according to performance metrics (mapk, apk, f1, precision, accuracy, etc.)
  
6. **painting_detector.py**
   - Utilizes the grabcut algorithm to remove backgrounds from paintings.

### Main Script:

- **main.py**
  - A main script used for testing, visualizing, and demonstrating the functionality of the system step by step.
