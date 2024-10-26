import cv2
import numpy as np

class PaintingDetector:
    def __init__(self, image):
        self.image = image
        self.mask = np.zeros(image.shape[:2], np.uint8)

    def detect_and_crop_paintings(self):
        self._apply_grabcut()
        binary_mask = self._create_binary_mask()
        cleaned_mask = self._clean_mask(binary_mask)
        contours = self._find_contours(cleaned_mask)
        cropped_paintings = self._crop_paintings(contours)
        return cleaned_mask, cropped_paintings

    def _apply_grabcut(self):
        rect = (10, 10, self.image.shape[1] - 20, self.image.shape[0] - 20)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(self.image, self.mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    def _create_binary_mask(self):
        mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        return self.image * mask2[:, :, np.newaxis]

    def _clean_mask(self, binary_mask):
        gray_result = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=3)
        kernel = np.ones((25, 25), np.uint8)
        binary_mask = cv2.erode(binary_mask, kernel, iterations=2)
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(binary_mask, kernel, iterations=5)

    def _find_contours(self, binary_mask):
        return cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    def _crop_paintings(self, contours):
        cropped_paintings = []
        for contour in contours:
            if cv2.contourArea(contour) < 1000:  # Filter small contours
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cropped = self.image[y:y+h, x:x+w]
            cropped_paintings.append(cropped)
        return cropped_paintings