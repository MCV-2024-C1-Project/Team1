import cv2
import glob
import os

class DataLoader:
    def __init__(self, args):

        """
        Initialize the DataLoader with a dictionary of arguments. 
        args = {dataset:"path to dataset"}
        """

        self.args = args

    def load_images_from_folder(self, extension="jpg", return_names=False):

        """
        Load all images from the folder.
        
        Returns:
            list: List of loaded images.
        """

        images = []
        if return_names:
            imagesNames = []
        for imagePath in glob.glob(self.args["dataset"] + f"/*.{extension}"):
            try:
                img = cv2.imread(imagePath)  # OpenCV reads images in BGR
                if img is not None:
                    images.append(img)
                    if return_names:
                        imagesNames.append(os.path.basename(imagePath))
            except Exception as e:
                print(f"Error loading image {imagePath}: {e}")
        
        if return_names:
            return images, imagesNames

        return images