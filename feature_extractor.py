import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, bins=256):
        """
        Initialize the FeatureExtractor with the number of bins for the histogram.
        
        Args:
            bins (int): Number of bins for the histogram.
        """
        self.bins = bins

    def resize_image(self, image, size=(256, 256)):
        return cv2.resize(image, size)
    
    def normalize_image(self, image):
        return image / 255.0
    
    # Functions for different color scales 
    def convert_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def convert_to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def convert_to_lab(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    def convert_to_ycrcb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    def convert_to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def compute_histogram(self, image, color_space='BGR', normalize=True):

        """
        Compute the histogram of the image in the specified color space.
        
        Args:
            image (numpy.ndarray): Input image.
            color_space (str): Color space to use ('Gray', 'RGB', 'BGR', 'HSV', 'LAB', 'YCrCb').
            normalize (bool): Whether to normalize the histogram.
        
        Returns:
            numpy.ndarray: Concatenated histogram for all channels.
        """

        def calc_hist(image, ranges):
             
            """
            Helper function to calculate the histogram for each channel and concatenate them.
            
            Args:
                image (numpy.ndarray): Input image.
                ranges (list of tuple): List of ranges for each channel.
            
            Returns:
                numpy.ndarray: Concatenated histogram for all channels.
            """
            
            hist = []
            for i in range(3):
                hist_channel = cv2.calcHist([image], [i], None, [self.bins], ranges[i])
                if normalize:
                    hist_channel = cv2.normalize(hist_channel, hist_channel).flatten()
                hist.append(hist_channel)

        if color_space == 'Gray':
            gray_image = self.convert_to_grayscale(image)
            hist = cv2.calcHist([gray_image], [0], None, [self.bins], [0, 256])
            if normalize:
                hist = cv2.normalize(hist, hist).flatten()
        
        elif color_space == 'RGB':
            rgb_image = self.convert_to_rgb(image)
            hist = calc_hist(image, [(0, 256), (0, 256), (0, 256)])
        
        elif color_space == 'BGR':
            hist = calc_hist(image, [(0, 256), (0, 256), (0, 256)])
        
        elif color_space == 'HSV':
            hsv_image = self.convert_to_hsv(image)
            hist = calc_hist(hsv_image, [(0, 180), (0, 256), (0, 256)])
        
        elif color_space == 'LAB':
            lab_image = self.convert_to_lab(image)
            hist = calc_hist(lab_image, [(0, 100), (-128, 127), (-128, 127)]) # if integer math is being used it is common to clamp a* and b* in the range of −128 to 127.
        
        elif color_space == 'YCrCb':
            ycrcb_image = self.convert_to_ycrcb(image)
            hist = calc_hist(ycrcb_image, [(0, 256), (0, 256), (0, 256)])
        
        else:
            raise ValueError("Unsupported color space")
        
        return hist


