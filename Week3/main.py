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

def get_predictions(imgs_query, imgs_ref, feature_methods=None, similarity_measure='MANHATTAN', k_best_results=1):
    """Get predicted matches between the query images and the reference database with flexible feature methods.

    Args:
        imgs_query (list): list of images used as query database
        imgs_ref (list): list of images used as reference database
        feature_methods (dict): Dictionary with feature extraction methods and specific parameters for each.
            Example:
            {
                'dct': {'block_size': 8, 'N': 6},
                'lbp': {'scales': [(1, 12)], 'block_size': 8},
                'wavelet': {'block_size': 8},
                'histogram': {'color_space': 'YCrCb', 'num_blocks': (10, 8)}
            }
        similarity_measure (str, optional): similarity metric to apply ('CORREL', 'CHISQR', 'INTRSC', 'HELLGR', 'MANHATTAN'). Defaults to 'MANHATTAN'.
        k_best_results (int, optional):  Number of most similar matches to return. Defaults to 1.

    Returns:
        list: list of top-k matches from the reference database, for each query image
    """
    descriptors_ref = []
    descriptors_query = []

    def extract_descriptors(img, feature_methods):
        """Extracts descriptors from an image according to specified methods.

        Args:
            img (numpy.ndarray): Image from which descriptors will be extracted.
            feature_methods (dict): Dictionary with feature methods and specific parameters for each. 
                Example:
                {
                    'dct': {'block_size': 8, 'N': 6},
                    'lbp': {'scales': [(1, 12)], 'block_size': 8},
                    'wavelet': {'block_size': 8},
                    'histogram': {'color_space': 'YCrCb', 'num_blocks': (10, 8)}
                }

        Returns:
            numpy.ndarray: Concatenated descriptors obtained from the image according to the specified methods.
        """
        descriptors = []
        for method, params in feature_methods.items():
            if method == 'dct':
                descriptors.append(FeatureExtractor().get_dct_descriptors(img, **params).astype(np.float64))
            elif method == 'lbp':
                descriptors.append(FeatureExtractor().get_lbp_descriptors(img, **params).astype(np.float64))
            elif method == 'wavelet':
                descriptors.append(FeatureExtractor().get_wavelet_descriptors(img, **params).astype(np.float64))
            elif method == 'histogram':
                blocks, block_histograms = FeatureExtractor(bins=32).divide_image_in_blocks(img, **params)
                histograms = np.concatenate(block_histograms)
                descriptors.append(histograms)
        return np.concatenate(descriptors)

    for img in imgs_ref:
        descriptors_ref.append(extract_descriptors(img, feature_methods))
    for img in imgs_query:
        descriptors_query.append(extract_descriptors(img, feature_methods))

    scores = SimilarityCalculator().compute_similarity(descriptors_query, descriptors_ref, similarity_measure)

    top_results = RetrievalSystem().retrieve_top_k(scores, reverse=False, k=k_best_results)
    
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
    # denoising.process_images(plot=False)

    imgs_denoised = DataLoader({"dataset": denoised_dir}).load_images_from_folder()

    # Get predictions
    feature_methods = {
        # 'dct': {'block_size': 8, 'N': 6},
        'lbp': {'scales': [(1, 12)], 'block_size': 16},
        # 'wavelet' : {'block_size': 8},
        # 'histogram': {'color_space': 'YCrCb', 'num_blocks': (10, 8)}
    }
    k_best_results = 1
    predictions = get_predictions(
                                    imgs_query = imgs_denoised, 
                                    imgs_ref = imgs_ref, 
                                    feature_methods = feature_methods, 
                                    similarity_measure = 'MANHATTAN', 
                                    k_best_results = k_best_results
                                )

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
    