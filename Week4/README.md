# Team 1 - MCV - C1 Project
[Final Presentation Link](https://docs.google.com/presentation/d/1Xz03CzWZmyQkbncYOwvQSF6lD40i6VKnCJMsriLYW8I/edit?usp=sharing)
## Table of Contents

1. [Context](#context)
2. [Artwork Recognition Project](#artwork-recognition-project)
3. [Project Overview](#project-overview)
   - [Final Objective](#final-objective)
   - [Week 4 Goal](#week-4-goal)
4. [Key Components](#key-components)
5. [Requirements](#requirements)
6. [Repository Structure](#repository-structure)
   - [Files](#files)
   - [Main Script](#main-script)

## Context

With the advancement of digital image processing technologies, the need for efficient methods to search and browse large image collections has grown. Traditional image retrieval methods can be classified into three main categories: **text-based**, **content-based**, and **semantic-based**. In everyday life, image searches are mostly conducted using search engines such as Google, which are primarily text-based. However, for art galleries and museums, visual content retrieval is crucial.

In this project, we focus on **content-based image retrieval (CBIR)** methods, specifically using low-level visual features such as histograms, keypoints, and descriptors. Our goal is to identify paintings from a database using these features, even when images are noisy or have varying color hues.

## Artwork Recognition Project

This project, developed over four weeks by **Team 1**, aims to create a model that can recognize a painting from a museum's database based on a given query image. Upon recognition, the system will return the name of the painting and the artist.

This project is developed using ![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

## Project Overview

The project is divided into four phases, with each phase focusing on a different component of building the artwork recognition model.

### Final Objective

- **Goal**: Given an image of a painting, the model will identify the corresponding artwork in the database and return the name of the painting and its artist.

## Week 4 Goal

- **Goal**: The project builds on previous work by utilizing advanced keypoint detectors and descriptors to find matches between the query and museum images. The steps for this week include:
  1. **Image Preprocessing**: Denoising and background removal from images with varying conditions (e.g., noise, color changes, multiple paintings per image).
  2. **Keypoint Detection and Descriptor Computation**: Using different keypoint detectors (e.g., Harris corner, DoG) and descriptors (SIFT, SURF, ORB, Color-SIFT).
  3. **Matching Descriptors**: The system will compare features between query and museum images to find potential matches. For this, the following descriptors are used:
     - **ORB**: ORiented FAST and Rotated BRIEF, covariant w.r.t. rotation and translation, not invariant to scaling.
     - **SIFT**: Scale-Invariant Feature Transform, robust to changes in scale, rotation, and illumination.
     - **Color-SIFT**: Similar to SIFT but with additional color channel information.
  4. **Evaluation**: Evaluate the model performance using **MAP@k** (Mean Average Precision at k).

### Key Components

1. **Image Preprocessing**:
   - **Denoising**: Remove noise from images using adaptive methods (e.g., Gaussian, bilateral, Wiener filtering).
   - **Background Removal**: Use the **GrabCut** algorithm to segment the foreground (paintings) from the background.
2. **Feature Extraction**:
   - **Keypoint Detection**: Detect keypoints using methods such as Harris corner detection, Difference of Gaussians (DoG), and Canny edge detection.
   - **Descriptors**:
     - **ORB**: Fast binary descriptors, robust to rotation and translation.
     - **SIFT**: Descriptors based on gradient orientation histograms, robust to scale and rotation.
     - **Color-SIFT**: Extended SIFT using color channels to enhance the ability to handle color variations.
3. **Matching**:
   - Use different similarity measures to compare descriptors and find the best matches between query and database images:
     - **Euclidean Distance**: Used for comparing continuous descriptors such as SIFT and Color-SIFT.
     - **Hamming Distance**: Used for comparing binary descriptors such as ORB.
4. **Evaluation**:
   - **MAP@k**: Mean Average Precision at k, used to assess the accuracy of the image retrieval system by comparing predicted matches with ground truth data.

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
pip install numpy opencv-python plotly scipy scikit-image pywavelets
```
You can install the required libraries with the following command:

```bash
pip install numpy opencv-python plotly scipy
````

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

8. **painting_detector.py**
   - Utilizes the grabcut algorithm to remove backgrounds from paintings.
9. **descriptors.py**
    - Descriptors class that computes keypoints and descriptors (ORB, SIFT, Color-SIFT) for a set of reference and query images, performs descriptor matching, and evaluates the best matches based on predefined thresholds and similarity measures.

### Main Script:

- **main.py**
  - A main script used for testing, visualizing, and demonstrating the functionality of the system step by step.
