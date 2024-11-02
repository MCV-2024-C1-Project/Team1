import cv2
import numpy as np
import os
from skimage import feature
def harris_corner_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    image[corners > 0.01 * corners.max()] = [0, 0, 255]  # Marcar esquinas en rojo
    return image





def sift_keypoint_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return image_with_keypoints

#to use if we denoise first
def canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)  # Ajusta los umbrales según sea necesario
    return edges
# not necessary to denoise if we adjust sigma


def skimage_canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convierte la imagen a escala de grises
    edges = feature.canny(gray, sigma=3, low_threshold=0.1, high_threshold=0.2)
    return edges

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(sobel_magnitude)  # Convertir a tipo de dato uint8
    return sobel_magnitude

# Ruta a la carpeta de imágenes
folder_path = r'C:\Users\Julia\OneDrive\Juliahacker\pythonProject1\Team1\content\qsd1_w4'

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)

        # Harris corner detection
        harris_result = harris_corner_detection(image.copy())

        # SIFT keypoints detection
        sift_result = sift_keypoint_detection(image.copy())

        # Canny edge detection
        canny_result_1 = canny_edge_detection(image.copy())
        #canny_result_2 = skimage_canny_edge_detection(image.copy())

        # Sobel edge detection
        sobel_result = sobel_edge_detection(image.copy())

        # Mostrar resultados
        cv2.imshow('Harris Corners', harris_result)
        cv2.imshow('SIFT Keypoints', sift_result)
        cv2.imshow('Canny Edges CV2', canny_result_1)
        #cv2.imshow('Canny Edges SKIMAGE', canny_result_1)
        cv2.imshow('Sobel Edges', sobel_result)
        cv2.waitKey(0)

cv2.destroyAllWindows()


