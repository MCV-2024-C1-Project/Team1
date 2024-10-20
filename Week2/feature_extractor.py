import cv2
import numpy as np
import matplotlib.pyplot as plt


class FeatureExtractor:
    def __init__(self, bins=256):
        """
        Initialize the FeatureExtractor with the number of bins for the histogram.
        
        Args:
            bins (int, optional): Number of bins for the histogram. Defaults to 256.
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
            color_space (str, optional): Color space to use ('Gray', 'RGB', 'BGR', 'HSV', 'LAB', 'YCrCb'). Defaults to BGR.
            normalize (bool, optional): Whether to normalize the histogram. Defaults to True.
            equalize (bool, optional): Whether to equalize the histogram (used only with Gray color space). Defaults to False.
        
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
            return hist

        if color_space == 'Gray':
            image = self.convert_to_grayscale(image)
            if equalize:
                image = cv2.equalizeHist(image)
            hist = cv2.calcHist([image], [0], None, [self.bins], [0, 256])
            if normalize:
                hist = cv2.normalize(hist, hist).flatten()
            return hist.ravel()

        elif color_space == 'RGB':
            image = self.convert_to_rgb(image)
            hist = calc_hist(image, [(0, 256), (0, 256), (0, 256)])
            hist = np.concatenate(hist)
        
        elif color_space == 'BGR':
            hist = calc_hist(image, [(0, 256), (0, 256), (0, 256)])
            hist = np.concatenate(hist)

        elif color_space == 'HSV':
            image = self.convert_to_hsv(image)
            hist = calc_hist(image, [(0, 180), (0, 256), (0, 256)])
            hist = np.concatenate(hist)

        elif color_space == 'LAB':
            image = self.convert_to_lab(image)
            hist = calc_hist(image, [(0, 100), (-128, 127), (-128, 127)]) # if integer math is being used it is common to clamp a* and b* in the range of −128 to 127.
            hist = np.concatenate(hist)

        elif color_space == 'YCrCb':
            image = self.convert_to_ycrcb(image)
            hist = calc_hist(image, [(0, 256), (0, 256), (0, 256)])
            hist = np.concatenate(hist)

        else:
            raise ValueError("Unsupported color space")

        return hist

    def divide_image_in_blocks(self, image, color_space='BGR', num_blocks=(5, 5)):
        '''
        Divide the image into a specified number of blocks (num_blocks), computes the histograms for the blocks and returns them.

        Args:
            image (numpy.ndarray): Input image.
            color_space (str, optional): Color space to use ('Gray', 'RGB', 'BGR', 'HSV', 'LAB', 'YCrCb'). Defaults to BGR.
            num_blocks (tuple, required): Number of blocks along (height, width). E.g., (5, 5) divides the image into 5 rows and 5 columns.

        Returns:
            image_blocks (list): A list of blocks that make up the image.
            block_histograms (list): A list of histograms, each corresponds to an image block.
        '''
        # Calc block size based on the image size and the number of blocks
        num_blocks_y, num_blocks_x = num_blocks
        block_size_y = image.shape[0] // num_blocks_y  # Block height
        block_size_x = image.shape[1] // num_blocks_x  # Block width
        
        blocks = []
        block_histograms = []
        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                # Calc start and end points
                y_start = i * block_size_y
                y_end = (i + 1) * block_size_y
                x_start = j * block_size_x
                x_end = (j + 1) * block_size_x

                # Include remaining pixels in the last block
                if i == num_blocks_y - 1:
                    y_end = image.shape[0]
                if j == num_blocks_x - 1:
                    x_end = image.shape[1]

                block = image[y_start:y_end, x_start:x_end]
                blocks.append(block)

                histogram = self.compute_histogram(block, color_space=color_space, normalize=True, equalize=False)
                block_histograms.append(histogram)

        return blocks, block_histograms
    
    
    def plot_histogram(self, hist, img=None):
        """Plot histogram

        Args:
            hist (list): list of histograms to plot
            img (numpy.ndarray, optional): image to be shown alongside histograms. Defaults to None.
        """
        n_col = len(hist)

        plt.figure()
        if img is not None:
            n_col+=1
            plt.subplot(1, n_col, 1)
            plt.imshow(img)
            plt.axis('off')
        
        for col in range(n_col-len(hist)+1, n_col+1):
            plt.subplot(1, n_col, col)
            plt.plot(hist[col-2])
        plt.show(block=False)

    def plot_histograms_divided(self,block_histograms):
        """
        Plot histograms of divided blocks

        Args:
            block_histograms: List of histograms to plot
        """
        for i, histogram in enumerate(block_histograms):
            try:
                plt.subplot(5, 5, i + 1)
                plt.plot(histogram)
                plt.title(f'Block {i + 1}')
                plt.axis('off')
            except ValueError:
                break

        plt.tight_layout()
        plt.show()

    def plot_rgb_visualized(self,image):
        """
        Plot image, RBG histogram and a line plot of the RBG values

        Args:
            image: Input image
        """
        plt.subplot(1, 3, 1)
        plt.axis("off")
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        blue_histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        red_histogram = cv2.calcHist([image], [1], None, [256], [0, 256])
        green_histogram = cv2.calcHist([image], [2], None, [256], [0, 256])

        plt.subplot(1, 3, 2)
        plt.title("Histogram of All Colors")
        plt.hist(blue_histogram, color="darkblue")
        plt.hist(green_histogram, color="green")
        plt.hist(red_histogram, color="red")

        plt.subplot(1, 3, 3)
        plt.title("Line Plots of All Colors")
        plt.plot(blue_histogram, color="darkblue")
        plt.plot(green_histogram, color="green")
        plt.plot(red_histogram, color="red")

        plt.tight_layout()
        plt.show()
