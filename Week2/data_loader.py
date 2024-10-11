import cv2
import glob

class DataLoader:
    def __init__(self, args):

        """
        Initialize the DataLoader with a dictionary of arguments. 
        args = {dataset:"path to dataset"}
        """

        self.args = args

    def load_images_from_folder(self):

        """
        Load all images from the folder.
        
        Returns:
            list: List of loaded images.
        """

        images = []
        for imagePath in glob.glob(self.args["dataset"] + "/*.jpg"):
            try:
                img = cv2.imread(imagePath)  # OpenCV reads images in BGR
                if img is not None:
                    images.append(img)
            except Exception as e:
                print(f"Error loading image {imagePath}: {e}")
        return images