import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from similarity_calculator import SimilarityCalculator
from retrieval_system import RetrievalSystem
from evaluator import Evaluator


def get_predictions(imgs_query, imgs_ref, color_space='HSV', similarity_measure='HELLGR', n_best_results=1, normalize_hist=True, equalize_hist=False):
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
    hist_ref = []
    hist_query = []
    for img in imgs_ref:
        hist_ref.append(FeatureExtractor().compute_histogram(img, color_space, normalize_hist, equalize_hist))
    for img in imgs_query:
        hist_query.append(FeatureExtractor().compute_histogram(img, color_space, normalize_hist, equalize_hist))

    # Compute similarity
    scores = SimilarityCalculator().compute_similarity(hist_query, hist_ref, similarity_measure)

    # Get predictions
    top_results = RetrievalSystem().retrieve_top_k(scores, reverse=False, k=n_best_results)
    
    return top_results

def main():
    # Load dataset
    imgs_ref = DataLoader({"dataset":"content/BBDD"}).load_images_from_folder()
    imgs_in = DataLoader({"dataset":"content/qsd1_w1"}).load_images_from_folder()

    # Get predictions
    k_best_results = 5
    predictions = get_predictions(imgs_query=imgs_in, imgs_ref=imgs_ref, color_space='YCrCb', similarity_measure='MANHATTAN', n_best_results=k_best_results, normalize_hist=True, equalize_hist=False)

    # Load ground-truth
    with open('content/qsd1_w1/gt_corresps.pkl', 'rb') as file:
        gt = pickle.load(file)

    # Evaluate
    mapk = Evaluator().mapk(gt, predictions, k_best_results)
    print(f"mapk: {mapk}")

    # Wait for user action to end
    # input("Press Enter to exit")




if __name__ == "__main__":
    main()
    
