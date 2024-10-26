import os
import cv2
import numpy as np
from scipy.signal import wiener
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from data_loader import DataLoader

class Denoising:
    def __init__(self, noisy_image_path, database_path, denoised_dir):
        self.noisy_image_path = noisy_image_path
        self.database_path = database_path
        self.denoised_dir = denoised_dir
        self.db_loader = DataLoader({"dataset": self.database_path})
        self.db_images = self.db_loader.load_images_from_folder(extension="jpg")
        self.mean_gradient, self.std_gradient = self.calculate_mean_gradient()
        self.threshold = self.mean_gradient - 2 * self.std_gradient

    def calculate_gradient(self, image):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return gradient_magnitude

    def calculate_mean_gradient(self):
        gradients = [self.calculate_gradient(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in self.db_images]
        mean_gradient = np.mean([np.mean(grad) for grad in gradients])
        std_gradient = np.std([np.mean(grad) for grad in gradients])
        return mean_gradient, std_gradient

    def has_noise(self, image):
        avg_gradient = np.mean(self.calculate_gradient(image))
        return avg_gradient > self.threshold

    def denoise_image(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def denoise_image_bilateral(self, image):
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    def denoise_image_wiener(self, image):
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Wiener filter
        denoised_image = wiener(gray_image, mysize=None, noise=None)
        # Convert back to BGR
        return cv2.cvtColor((denoised_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def calculate_statistics(self, original, denoised):
        psnr_value = cv2.PSNR(original, denoised)
        ssim_value = ssim(original, denoised, multichannel=True)
        return psnr_value, ssim_value

    def process_images(self, plot=False):
        noisy_loader = DataLoader({"dataset": self.noisy_image_path})
        noisy_images, noisy_names = noisy_loader.load_images_from_folder(extension="jpg", return_names=True)

        print(f'Mean Gradient from Database: {self.mean_gradient}')
        print(f'Standard Deviation of Gradient from Database: {self.std_gradient}')
        print(f'Threshold for Noise Detection: {self.threshold}')

        for idx, image in enumerate(noisy_images):
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Determine if the image has noise
            if self.has_noise(gray_image):  # Usar el umbral
                denoised_image = self.denoise_image(image)
            else:
                denoised_image = image

            # Calc statistics
            psnr_value, ssim_value = self.calculate_statistics(gray_image,
                                                               cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY))

            # Save denoised images
            cv2.imwrite(os.path.join(self.denoised_dir, noisy_names[idx]), denoised_image)

            if plot:
                # Show original and denoised
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title('Original Image')
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title(f'Denoised Image\nPSNR: {psnr_value:.2f}, SSIM: {ssim_value:.2f}')
                plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
                plt.axis('off')

                plt.show()


# Uso de la clase Denoising
# noisy_images_path = "../content/qsd1_w3_test"
# database_path = "../content/BBDD_test"

# output_dir = "output"
# denoised_dir = os.path.join(output_dir, "denoised")
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(denoised_dir, exist_ok=True)

# denoising = Denoising(noisy_images_path, database_path, denoised_dir)
# denoising.process_images()

