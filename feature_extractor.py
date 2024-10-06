import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    
    def compute_histogram(self, image, color_space='BGR', normalize=True, equalize=False):
        """
        Compute the histogram of the image in the specified color space.
        
        Args:
            image (numpy.ndarray): Input image.
            color_space (str): Color space to use ('Gray', 'RGB', 'BGR', 'HSV', 'LAB', 'YCrCb').
            normalize (bool): Whether to normalize the histogram.
        
        Returns:
            numpy.ndarray: Concatenated histogram for all channels.
        """

        color_space_conversion = {
            'Gray': (self.convert_to_grayscale, [(0, 256)]),
            'RGB': (self.convert_to_rgb, [(0, 256), (0, 256), (0, 256)]),
            'BGR': (lambda x: x, [(0, 256), (0, 256), (0, 256)]),
            'HSV': (self.convert_to_hsv, [(0, 180), (0, 256), (0, 256)]),
            'LAB': (self.convert_to_lab, [(0, 100), (-128, 127), (-128, 127)]),
            'YCrCb': (self.convert_to_ycrcb, [(0, 256), (0, 256), (0, 256)])
        }

        if color_space not in color_space_conversion:
            raise ValueError("Unsupported color space")

        convert_func, ranges = color_space_conversion[color_space]
        image = convert_func(image)
        
        if color_space == 'Gray':
            if equalize:
                image = cv2.equalizeHist(image)
            hist = cv2.calcHist([image], [0], None, [self.bins], [0, 256])
            if normalize:
                hist = cv2.normalize(hist, hist).flatten()
            return hist
        
        hist = []
        for i in range(3):
            hist_channel = cv2.calcHist([image], [i], None, [self.bins], ranges[i])
            if normalize:
                hist_channel = cv2.normalize(hist_channel, hist_channel).flatten()
            hist.append(hist_channel)
        
        return np.concatenate(hist)
    
    def plot_histogram(self, hist_dict, img=None):
        """Plot histogram

        Args:
            hist_dict (dict): dictionary of histograms to plot.
            img (numpy.ndarray, optional): image to be shown alongside histograms. Defaults to None.
        """
        n_col = len(hist_dict)

        plt.figure(figsize=(15, 5))
        if img is not None:
            n_col += 1
            plt.subplot(1, n_col, 1)
            plt.imshow(img)
            plt.axis('off')

        for i, (title, hist) in enumerate(hist_dict.items(), start=2 if img is not None else 1):
            plt.subplot(1, n_col, i)
            plt.plot(hist, linewidth=2)
            plt.title(title)
            plt.grid(True)
        plt.show(block=False)

def store_vectors(folder_path, extractor, output_file='features.npy'):
    """
    Store images as vectors in a csv file
    
    Args:
        folder_path: folder with images
        extractor: An instance of the FeatureExtractor object
        output_file: file that stores the feature vectors in a numpy array

    """
    data_loader = DataLoader(folder_path)
    images, filenames = data_loader.load_images_from_folder()

    feature_data = [] # Store index, filename, color scale, and histogram
    
    # Loop over each image and compute its feature vector
    for index, image in enumerate(images):
        feature_vector = extractor.compute_histogram(image, color_space)  # Extract features with specified color scale
        
        feature_data.append((index, filenames[index], color_space, feature_vector))

    # Save data as a NumPy array
    np.save(output_file, feature_data)

    print(f"Feature vectors saved to {output_file}.")


