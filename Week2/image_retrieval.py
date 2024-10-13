import os
import sys
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add Week1 to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Week1')))
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from similarity_calculator import SimilarityCalculator
from retrieval_system import RetrievalSystem
from evaluator import Evaluator


def get_predictions(imgs_query, imgs_ref, color_space='HSV', bins=256, similarity_measure='HELLGR', n_best_results=1, normalize_hist=True, equalize_hist=False):
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
    n_blocks_rows = 10
    n_blocks_cols = 8
    # Get image descriptors
    hist_ref = []
    hist_query = []
    for img in imgs_ref:
        # Get descriptors
        blocks, block_histograms = FeatureExtractor(bins=32).divide_image_in_blocks(img, color_space="YCrCb", num_blocks=(n_blocks_rows, n_blocks_cols))
        concatenated_histogram = np.concatenate(block_histograms)
        hist_ref.append(concatenated_histogram)
    for img in imgs_query:
        blocks, block_histograms = FeatureExtractor(bins=32).divide_image_in_blocks(img, color_space="YCrCb", num_blocks=(n_blocks_rows, n_blocks_cols))
        concatenated_histogram = np.concatenate(block_histograms)
        hist_query.append(concatenated_histogram)
    
    # Compute similarity
    scores = SimilarityCalculator().compute_similarity(hist_query, hist_ref, similarity_measure)

    # Get predictions
    top_results = RetrievalSystem().retrieve_top_k(scores, reverse=False, k=n_best_results)
    
    return top_results
