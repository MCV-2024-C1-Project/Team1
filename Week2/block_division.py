import cv2
import numpy as np
import matplotlib.pyplot as plt

class BlockBasedHistogram:
    def __init__(self, image, block_size=(32, 32), bins=64, normalize=True):
        """
        Initialize the BlockBasedHistogram class.

        Args:
            image (numpy.ndarray): Input image.
            block_size (tuple, optional): Size of the blocks. Defaults to (32, 32).
            bins (int, optional): Number of bins for the histogram. Defaults to 64.
            normalize (bool, optional): Whether to normalize the histograms. Defaults to True.
        """
        self.image = image
        self.block_size = block_size
        self.bins = bins
        self.normalize = normalize
    
    def compute_block_histogram(self, block, color_space, histogram_type='2D', combine_hs_ycrcb=False):
        """
        Compute the histogram of the block in the specified color space.

        Args:
            block (numpy.ndarray): Image block.
            color_space (str): Color space to use ('gray', 'hsv', 'YCrCb').
            histogram_type (str): Type of histogram ('1D', '2D', or '3D').
            combine_hs_ycrcb (bool): Whether to combine HS and CrCb histograms.

        Returns:
            numpy.ndarray: Flattened histogram for the block.
        """
        hist = []

        if color_space == 'gray':
            histogram = cv2.calcHist([block], [0], None, [self.bins], [0, 256])
            if self.normalize:
                cv2.normalize(histogram, histogram)
            return histogram.flatten()

        if histogram_type == '1D':
            if color_space == 'hsv':
                hsv_block = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
                for ch in range(3):  # H, S, V channels
                    hist_ch = cv2.calcHist([hsv_block], [ch], None, [self.bins], [0, 256 if ch != 0 else 180])
                    if self.normalize:
                        cv2.normalize(hist_ch, hist_ch)
                    hist.extend(hist_ch.flatten())

            elif color_space == 'YCrCb':
                ycrcb_block = cv2.cvtColor(block, cv2.COLOR_BGR2YCrCb)
                for ch in range(3):  # Y, Cr, Cb channels
                    hist_ch = cv2.calcHist([ycrcb_block], [ch], None, [self.bins], [0, 256])
                    if self.normalize:
                        cv2.normalize(hist_ch, hist_ch)
                    hist.extend(hist_ch.flatten())

        elif histogram_type == '2D':
            if color_space == 'hsv':
                hsv_block = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
                hist_hs = cv2.calcHist([hsv_block], [0, 1], None, [self.bins, self.bins], [0, 180, 0, 256])
                if self.normalize:
                    cv2.normalize(hist_hs, hist_hs)
                hist.extend(hist_hs.flatten())

            elif color_space == 'YCrCb':
                ycrcb_block = cv2.cvtColor(block, cv2.COLOR_BGR2YCrCb)
                hist_ycr = cv2.calcHist([ycrcb_block], [1, 2], None, [self.bins, self.bins], [0, 256, 0, 256])
                if self.normalize:
                    cv2.normalize(hist_ycr, hist_ycr)
                hist.extend(hist_ycr.flatten())

        elif histogram_type == '3D':
            if color_space == 'hsv':
                hsv_block = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
                hist_hsv = cv2.calcHist([hsv_block], [0, 1, 2], None, [self.bins, self.bins, self.bins], [0, 180, 0, 256, 0, 256])
                if self.normalize:
                    cv2.normalize(hist_hsv, hist_hsv)
                hist.extend(hist_hsv.flatten())

            elif color_space == 'YCrCb':
                ycrcb_block = cv2.cvtColor(block, cv2.COLOR_BGR2YCrCb)
                hist_ycrcb = cv2.calcHist([ycrcb_block], [0, 1, 2], None, [self.bins, self.bins, self.bins], [0, 256, 0, 256, 0, 256])
                if self.normalize:
                    cv2.normalize(hist_ycrcb, hist_ycrcb)
                hist.extend(hist_ycrcb.flatten())

        if combine_hs_ycrcb and color_space in ['hsv', 'YCrCb']:
            if color_space == 'hsv':
                hist_combined = self.compute_block_histogram(block, 'YCrCb', histogram_type='2D')
                hist.extend(hist_combined)

        return np.array(hist).flatten()
    
    def divide_into_blocks_and_compute_histograms(self, color_space='gray', histogram_type='2D', combine_hs_ycrcb=False):
        """
        Divide the image into blocks and compute histograms for each block.

        Args:
            color_space (str, optional): Color space to use ('gray', 'hsv', 'YCrCb'). Defaults to 'gray'.
            histogram_type (str): Type of histogram ('1D', '2D', or '3D').
            combine_hs_ycrcb (bool): Whether to combine HS and CrCb histograms.

        Returns:
            list: List of tuples, each containing a block and its histogram.
        """
        height, width = self.image.shape[:2]
        
        # Check if block size is the same as image size
        if (self.block_size[0] == height) and (self.block_size[1] == width):
            block = self.image
            block_histogram = self.compute_block_histogram(block, color_space, histogram_type, combine_hs_ycrcb)
            return [(block, block_histogram)]
        
        h_blocks = height // self.block_size[0]
        w_blocks = width // self.block_size[1]

        block_data = []

        for i in range(h_blocks):
            for j in range(w_blocks):
                block = self.image[i*self.block_size[0]:(i+1)*self.block_size[0], j*self.block_size[1]:(j+1)*self.block_size[1]]
                block_histogram = self.compute_block_histogram(block, color_space, histogram_type, combine_hs_ycrcb)
                block_data.append((block, block_histogram))

        return block_data
    
    def plot_histograms_with_blocks(self, block_data, max_blocks=10):
        num_blocks = min(len(block_data), max_blocks)
        if num_blocks == 0:
            return  # No blocks to plot

        # Set up the figure and axes
        if num_blocks > 1:
            fig, axs = plt.subplots(2, num_blocks, figsize=(20, 5))
        else:
            fig, axs = plt.subplots(2, 1, figsize=(10, 5))

        for idx, (block, hist) in enumerate(block_data[:num_blocks]):
            if num_blocks > 1:
                axs[0, idx].imshow(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))
                axs[0, idx].axis('off')
                axs[0, idx].set_title(f'Block {idx + 1}')
                axs[1, idx].plot(hist)
                axs[1, idx].set_xlim([0, len(hist)])
                axs[1, idx].set_title(f'Histogram {idx + 1}')
            else:  # Only one block
                axs[0].imshow(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))
                axs[0].axis('off')
                axs[0].set_title('Block 1')
                axs[1].plot(hist)
                axs[1].set_xlim([0, len(hist)])
                axs[1].set_title('Histogram 1')

        plt.tight_layout()
        plt.show()
    
    def plot_combined_histogram(self, concatenated_histograms):
        """
        Plot the concatenated histograms of all blocks.

        Args:
            concatenated_histograms (numpy.ndarray): Concatenated histograms.
        """
        plt.figure(figsize=(10, 4))
        plt.title("Concatenated Histograms of Image Blocks")
        plt.xlabel("Bin")
        plt.ylabel("Frequency")
        plt.plot(concatenated_histograms)
        plt.show()

    def visualize_image_with_blocks(self):
        """
        Visualize the image with block divisions.
        """
        img_copy = self.image.copy()
        height, width = img_copy.shape[:2]
        
        # Draw horizontal lines
        for i in range(0, height, self.block_size[0]):
            cv2.line(img_copy, (0, i), (width, i), (255, 255, 255), 2)  # White lines, thickness 2
        
        # Draw vertical lines
        for j in range(0, width, self.block_size[1]):
            cv2.line(img_copy, (j, 0), (j, height), (255, 255, 255), 2)  # White lines, thickness 2
        
        plt.figure(figsize=(10, 10))
        plt.title("Image with Block Divisions")
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def process_and_visualize(self, color_space='gray', histogram_type='2D', combine_hs_ycrcb=False,plot=False):
        """
        Process the image and visualize the histograms.

        Args:
            color_space (str, optional): Color space to use ('gray', 'hsv', 'YCrCb'). Defaults to 'gray'.
            histogram_type (str): Type of histogram ('1D', '2D', or '3D').
            combine_hs_ycrcb (bool): Whether to combine HS and CrCb histograms.
            plot (bool): whether to plot or not. Defaults to False
        """
        if plot:
            self.visualize_image_with_blocks()
        block_data = self.divide_into_blocks_and_compute_histograms(color_space, histogram_type, combine_hs_ycrcb)
        
        # Visualize individual histograms with corresponding blocks
        if plot:
            self.plot_histograms_with_blocks(block_data)
        
        # Concatenate histograms
        concatenated_histograms = np.hstack([hist for _, hist in block_data])
        if plot:
            # Visualize concatenated histogram
            self.plot_combined_histogram(concatenated_histograms)
        return concatenated_histograms
