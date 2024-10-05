import cv2
import matplotlib.pyplot as plt
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from similarity_calculator import SimilarityCalculator


def main():
    # Load dataset
    bbdd = DataLoader({"dataset":"content/BBDD"}).load_images_from_folder()
    qst1_w4 = DataLoader({"dataset":"content/qst1_w4"}).load_images_from_folder()

    # Extract features
    # Compute histogram
    ft = FeatureExtractor()
    hist_eq_bbdd = []
    hist_bbdd = []
    for img in bbdd:
        # Histogram equalization
        hist_eq_bbdd.append(ft.compute_histogram(img, color_space='Gray', normalize=False, equalize=True))
        # HSV histogram
        hist_bbdd.append(ft.compute_histogram(img, color_space='HSV'))

    hist_eq_qst1_w4 = []
    hist_qst1_w4 = []
    for img in qst1_w4:
        # Histogram equalization
        hist_eq_qst1_w4.append(ft.compute_histogram(img, color_space='Gray', normalize=False, equalize=True))
        # HSV histogram
        hist_qst1_w4.append(ft.compute_histogram(img, color_space='HSV'))


    # Wait for user action to end
    input("Press Enter to exit")



if __name__ == "__main__":
    main()
    
