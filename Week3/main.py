import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from similarity_calculator import SimilarityCalculator
from retrieval_system import RetrievalSystem
from evaluator import Evaluator
from denoising import Denoising


def get_predictions(imgs_query, imgs_ref, color_space='HSV', similarity_measure='HELLGR', n_best_results=1, normalize_hist=True, equalize_hist=False, block_size_dct=8, N_dct=6):
    """Get predicted matches between the query images and the reference database

    Args:
        imgs_query (list): list of images used as query database
        imgs_ref (list): list of images used as reference database
        color_space (str, optional): color space in which to compute the histogram ('Gray', 'RGB', 'BGR', 'HSV', 'LAB', 'YCrCb'). Defaults to 'HSV'.
        similarity_measure (str, optional): similarity metric to apply ('CORREL', 'CHISQR', 'INTRSC', 'HELLGR'). Defaults to 'HELLGR'.
        n_best_results (int, optional):  Number of most similar matches to return. Defaults to 1.
        normalize_hist (bool, optional): Wheter to normalize histograms. Defaults to True.
        equalize_hist (bool, optional): Wheter to equalize histograms (used only with Gray color space). Defaults to False.

    Returns:
        list: list of best n matches from the reference database, for each query image
    """
    # Get image descriptors
    descriptors_ref = []
    descriptors_query = []
    for img in imgs_ref:
        descriptors_ref.append(FeatureExtractor().get_dct_descriptors(img, block_size=8, N=6))
    for img in imgs_query:
        descriptors_query.append(FeatureExtractor().get_dct_descriptors(img, block_size=8, N=6))
    
    # Compute similarity
    scores = SimilarityCalculator().compute_similarity(descriptors_query, descriptors_ref, similarity_measure)

    # Get predictions
    top_results = RetrievalSystem().retrieve_top_k(scores, reverse=False, k=n_best_results)
    
    return top_results

def main():
    imgs_ref_path = "../content/BBDD"
    imgs_non_aug_path = "../content/qsd1_w3/non_augmented"
    imgs_noisy_path = "../content/qsd1_w3"

    output_dir = "output"
    denoised_dir = os.path.join(output_dir, "denoised")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(denoised_dir, exist_ok=True)

    # Load dataset
    imgs_ref = DataLoader({"dataset":imgs_ref_path}).load_images_from_folder()
    imgs_non_aug = DataLoader({"dataset":imgs_non_aug_path}).load_images_from_folder()
    imgs_noisy = DataLoader({"dataset":imgs_noisy_path}).load_images_from_folder()

    # Denoise
    # denoising = Denoising(imgs_noisy_path, imgs_ref_path, denoised_dir)
    # denoising.process_images()

    imgs_denoised = DataLoader({"dataset":denoised_dir}).load_images_from_folder()

    # Get predictions
    k_best_results = 5
    predictions = get_predictions(imgs_query=imgs_denoised, imgs_ref=imgs_ref, color_space='YCrCb', similarity_measure='MANHATTAN', n_best_results=k_best_results, normalize_hist=True, equalize_hist=False)

    # Load ground-truth
    with open('../content/qsd1_w3/gt_corresps.pkl', 'rb') as file:
        gt = pickle.load(file)

    print(f"\nlen predictions: {len(predictions)}")
    print(predictions)
    print(f"\nlen gt: {len(gt)}")
    print(gt)

    # Evaluate
    mapk = Evaluator().mapk(gt, predictions, k_best_results)
    print(f"mapk: {mapk}")

    # Wait for user action to end
    # input("Press Enter to exit")


if __name__ == "__main__":
    main()
    
