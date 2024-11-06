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
    top_results = []

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
    
    if all(isinstance(i, list) for i in imgs_query):  # List of lists
        for img_group in imgs_query:
            # print(f"img group: {len(img_group)}")
            group_descriptors = []

            for img in img_group:
                group_descriptors.append(extract_descriptors(img, feature_methods))
            # print(f"group descriptors: {len(group_descriptors)}")

            group_scores = []
            for descriptor in group_descriptors:
                # print(f"descriptor: {len(descriptor)}")
                # print(f"descriptor_ref: {len(descriptors_ref)}")
                scores = SimilarityCalculator().compute_similarity([descriptor], descriptors_ref, similarity_measure)
                group_scores.append(scores)
            # print(f"group scores: {len(group_scores)}")
            
            group_top_results = [RetrievalSystem().retrieve_top_k(score, reverse=False, k=k_best_results)[0] for score in group_scores]
            # print(f"group top results: {len(group_top_results)}")

            group_top_results = [result[0] if len(result) == 1 else result for result in group_top_results]

            top_results.append(group_top_results)
    else:
        for img in imgs_query:
            descriptors_query.append(extract_descriptors(img, feature_methods))

        scores = SimilarityCalculator().compute_similarity(descriptors_query, descriptors_ref, similarity_measure)

        top_results = RetrievalSystem().retrieve_top_k(scores, reverse=False, k=k_best_results)
    
    return top_results


def main():
    # Load reference dataset
    imgs_ref_path = "C:/Users/laila/Downloads/BBDD/BBDD"
    imgs_ref = DataLoader({"dataset":imgs_ref_path}).load_images_from_folder()

    # TASK 1 ----------------------------------------------------------------
    # Load datasets
    # Gt images
    imgs_non_aug_path = "C:/Users/laila/Downloads/qsd1_w3/qsd1_w3/non_augmented"
    imgs_non_aug = DataLoader({"dataset":imgs_non_aug_path}).load_images_from_folder()
    # Noisy images
    imgs_noisy_path = "C:/Users/laila/Downloads/qsd1_w3/qsd1_w3"
    imgs_noisy, noisy_names = DataLoader({"dataset":imgs_noisy_path}).load_images_from_folder(return_names=True)

    output_dir = "output"
    denoised_dir = os.path.join(output_dir, "denoised")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(denoised_dir, exist_ok=True)

    imgs_denoised = Denoising(imgs_noisy, imgs_ref, noisy_names=noisy_names, denoised_dir=denoised_dir).process_images(plot=False)

    print(len(imgs_denoised))

    # --------------------------------------------------------------------------------

    # TASK 2 -------------------------------------------------------------------------
    print("TASK 2 -------------------------------------------------")
    imgs_denoised = DataLoader({"dataset": denoised_dir}).load_images_from_folder()

    # Get predictions
    feature_methods = {
        'dct': {'block_size': 8, 'N': 6},
        'lbp': {'scales': [(1, 12)], 'block_size': 8, 'method': "uniform"},
        'wavelet' : {'block_size': 8},
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
    with open('C:/Users/laila/Downloads/qsd1_w3/qsd1_w3/gt_corresps.pkl', 'rb') as file:
        gt = pickle.load(file)

    print(f"\nlen predictions: {len(predictions)}")
    print(predictions)
    print(f"\nlen gt: {len(gt)}")
    print(gt)

    # Evaluate
    mapk = Evaluator().mapk(gt, predictions, k_best_results)
    print(f"mapk: {mapk}")
    print("--------------------------------------------------------")
    # --------------------------------------------------------------------------------

    # TASK 3 + TASK 4 ----------------------------------------------------------------
    print("TASK 3 + TASK 4 ----------------------------------------")

    with open('Team1/Week3/cropped_paintings.pickle', 'rb') as file:
        imgs_cropped = pickle.load(file)
    
    # Denoise
    imgs_denoised = Denoising(imgs_cropped, imgs_ref).process_images(plot=False)

    # Get predictions
    feature_methods = {
        'dct': {'block_size': 8, 'N': 6},
        'lbp': {'scales': [(1, 12)], 'block_size': 8, 'method': "uniform"},
        'wavelet' : {'block_size': 8},
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
    with open('C:/Users/laila/Downloads/qsd2_w3/qsd2_w3/gt_corresps.pkl', 'rb') as file:
        gt = pickle.load(file)

    print(f"\nlen predictions: {len(predictions)}")
    print(predictions)
    print(f"\nlen gt: {len(gt)}")
    print(gt)

    # Evaluate
    mapk = Evaluator().mapk(gt, predictions, k_best_results)
    print(f"mapk: {mapk}")

    print("--------------------------------------------------------")

    # Wait for user action to end
    # input("Press Enter to exit")


if __name__ == "__main__":
    main()
    