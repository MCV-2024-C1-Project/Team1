import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct


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
    
    def divide_img_in_n_blocks(self, img, n_blocks=(4,4), block_size=None, plot=False):
        if img.ndim > 2:
            h, w, ch = img.shape
        else:
            h, w = img.shape
            ch = 1

        if block_size is None:
            n_col, n_row = n_blocks
            block_h = h // n_row
            block_w = w // n_col
        else:
            block_h, block_w = block_size, block_size if isinstance(block_size, int) else block_size
            n_row = h // block_h
            n_col = w // block_w
        
        # Adjust image to fit an exact number of blocks (discard remaining pixels from image)
        img_adj = img[:block_h*n_row, :block_w*n_col]

        # Divide image
        # Dividir en bloques (subarrays)
        blocks = (img_adj
                .reshape(n_row, block_h, n_col, block_w, ch)
                .swapaxes(1, 2))
        
        if plot:
            self.plot_img_divided_n_blocks(blocks, (n_col, n_row))

        return blocks


    def get_dct_descriptors(self, img, n_blocks=(4,4), block_size=None, N=6):

        def compute_dct(blocks, axis_h=2, axis_w=3, norm='ortho'):
            # Apply DCT on block height
            dct_blocks = dct(blocks, axis=axis_h, norm=norm)
            # Apply DCT on block width
            dct_blocks = dct(dct_blocks, axis=axis_w, norm=norm)
            return dct_blocks

        def zigzag_indices(block_h, block_w):
            # Create matrix of indices
            indices = np.indices((block_h, block_w)).reshape(2, -1).T
            # Sum indices to determine the zig-zag order
            indices_sum = indices[:, 0] + indices[:, 1]
            
            # Order indices first by sum of indices (in case of repeated sum, ordered by column)
            zigzag_order = indices[np.lexsort((indices[:, 1], indices_sum))]
            sorted_indices = np.ravel_multi_index(zigzag_order.T, (block_h, block_w))

            return sorted_indices[:N]

        def select_coefficients():
            #  Zig-zag scan and select first N coefficients per block
            zigzag_idx = zigzag_indices(block_h, block_w)
            selected_coefficients = np.zeros((n_row, n_col, N, ch))
            for i in range(n_row):
                for j in range(n_col):
                    for c in range(ch):
                        _blocks_dct = blocks_dct[i, j, :, :, c]
                        selected_coefficients[i, j, :, c] = _blocks_dct.flatten()[zigzag_idx]
                        
            # Re-orginize and concatenate array by channels: (rows, cols, N, ch) --> (ch, rows, cols, N)
            coefficients_by_ch = np.moveaxis(selected_coefficients, -1, 0)
            coefficients_concat = np.concatenate(coefficients_by_ch, axis=1)
            
            return coefficients_concat.flatten()

        def get_unidim_idx(i, j, k, c, n_blocks_h, n_blocks_w, N):
            return (c * (n_blocks_h * n_blocks_w * N)) + (i * n_blocks_w * N) + (j * N) + k

        # Convert image to grayscale
        # img = self.convert_to_grayscale(img)
        # img = self.convert_to_ycrcb(img)
        # img = img[:,:,0]
        
        # Resize image
        resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        
        # Divide image in blocks
        if block_size is not None:
            blocks = self.divide_img_in_n_blocks(img=resized_img, block_size=block_size)
            block_h, block_w = block_size, block_size if isinstance(block_size, int) else block_size
        else:
            blocks = self.divide_img_in_n_blocks(img=resized_img, n_blocks=n_blocks)
            block_h, block_w = blocks.shape[2], blocks.shape[3]
        
        n_col, n_row, ch = blocks.shape[1], blocks.shape[0], blocks.shape[4]

        # Compute DCT for each block
        blocks_dct = compute_dct(blocks)
        dct_descriptors = select_coefficients()

        # Normalization
        mean_global = np.mean(dct_descriptors)
        std_global = np.std(dct_descriptors) if np.std(dct_descriptors) != 0 else 1
        dct_descriptors = (dct_descriptors - mean_global) / std_global

        return dct_descriptors

    
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

    def plot_img_divided_n_blocks(self, blocks, n_blocks=(4, 4)):
        n_col, n_row = n_blocks
        fig, axes = plt.subplots(n_row, n_col, figsize=(10, 10))
        for i in range(n_row):
            for j in range(n_col):
                axes[i, j].imshow(blocks[i, j])
                axes[i, j].axis('off')
        plt.show(block=False)