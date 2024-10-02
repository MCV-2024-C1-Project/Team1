import cv2
import os

class DataLoader:
    def __init__(self, folder_path):

        """
        Initialize the DataLoader with a folder containing images.
        """

        self.folder_path = folder_path

    def load_images_from_folder(self):

        """
        Load all images from the folder.
        
        Returns:
            list: List of loaded images.
        """

        images = []
        for filename in os.listdir(self.folder_path):
            try:
                img = cv2.imread(os.path.join(self.folder_path, filename))  # OpenCV reads images in BGR
                if img is not None:
                    images.append(img)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
        return images

    

