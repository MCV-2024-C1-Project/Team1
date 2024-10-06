import cv2
from scipy.spatial import distance as dist

class SimilarityCalculator:

    def compute_similarity(self, dict_images_hist, methodName="Correlation"):
        
        """
        Compute the similarity between histograms in the specified color space.
        
        Args:
            dict_images_hist (list of dictionaries): A list containing 2 images and their corresponding dictionary of histograms.
            method (str): Which method to use.
        
        Returns:
            results (dict): A dictionary of similarities.
        """

        # Define available methods
        OPENCV_METHODS = {
            "Correlation": cv2.HISTCMP_CORREL,
            "Chi-Squared": cv2.HISTCMP_CHISQR,
            "Intersection": cv2.HISTCMP_INTERSECT,
            "Hellinger": cv2.HISTCMP_BHATTACHARYYA
        }

        SCIPY_METHODS = {
            "Euclidean": dist.euclidean,
            "Manhattan": dist.cityblock,
            "Chebyshev": dist.chebyshev
        }

        # METHOD #1: UTILIZING OPENCV
        if methodName in OPENCV_METHODS:
            
            method = OPENCV_METHODS[methodName]
            # initialize the results dictionary and the sort
            # direction
            results = {}
            reverse = False
            # if we are using the correlation or intersection
            # method, then sort the results in reverse order
            if methodName in ("Correlation", "Intersection"):
                    reverse = True

            # loop over the dictionary 
            for color_scale in dict_images_hist[0]:
                # compute the distance between the two histograms
                # using the method and update the results dictionary
                distance = cv2.compareHist(dict_images_hist[0][color_scale], dict_images_hist[1][color_scale], method)
                results[color_scale] = distance
            # sort the results
            results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)

        # METHOD #2: UTILIZING SCIPY
        elif methodName in SCIPY_METHODS:

            method = SCIPY_METHODS[methodName]
            # initialize the dictionary 
            results = {}
            # loop over the index
            for color_scale in dict_images_hist[0]:
                    # compute the distance between the two histograms
                    # using the method and update the results dictionary
                    distance = method(dict_images_hist[0][color_scale], dict_images_hist[1][color_scale])
                    results[color_scale] = distance
                # sort the results
            results = sorted([(v, k) for (k, v) in results.items()])

        

        return results 