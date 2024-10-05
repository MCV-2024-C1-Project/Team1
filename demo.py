import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from similarityCalculator import SimilarityCalculator


def main():
    # Load dataset
    imgs_ref = DataLoader({"dataset":"content/BBDD"}).load_images_from_folder()
    imgs_in = DataLoader({"dataset":"content/qsd1_w1"}).load_images_from_folder()

    # Extract features
    # Compute histogram
    ft = FeatureExtractor()

    hist_ref = []
    hist_eq_ref = []
    for img in imgs_ref:
        # HSV histogram
        hist_ref.append(ft.compute_histogram(img, color_space='HSV'))
        # Histogram equalization
        hist_eq_ref.append(ft.compute_histogram(img, color_space='Gray', normalize=False, equalize=True))

    hist_in = []
    hist_eq_in = []
    for img in imgs_in:
        # HSV histogram
        hist_in.append(ft.compute_histogram(img, color_space='HSV'))
        # Histogram equalization
        hist_eq_in.append(ft.compute_histogram(img, color_space='Gray', normalize=False, equalize=True))

    sc = SimilarityCalculator()
    result = sc.compute_similarity(hist_in, hist_ref, "CHISQR")
    print(result.shape)

    # Wait for user action to end
    input("Press Enter to exit")


if __name__ == "__main__":
    main()
    
