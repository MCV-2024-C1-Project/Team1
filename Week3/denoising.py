import os
import cv2
import numpy as np
from scipy.signal import wiener
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from data_loader import DataLoader

class Denoising:
    def __init__(self, noisy_images, reference_images, noisy_names=None, denoised_dir=None):
        self.noisy_images = noisy_images
        self.noisy_names = noisy_names
        self.db_images = reference_images
        self.denoised_dir = denoised_dir
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

    def process_images(self, denoised_path=None, plot=False):
        print(f'Mean Gradient from Database: {self.mean_gradient}')
        print(f'Standard Deviation of Gradient from Database: {self.std_gradient}')
        print(f'Threshold for Noise Detection: {self.threshold}')
        denoised_images = []

        def process_single_image(image, idx):
            """Helper function to process a single image."""
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Determine if the image has noise
            if self.has_noise(gray_image):  # Use threshold
                denoised_image = self.denoise_image(image)
            else:
                denoised_image = image

            # Calculate statistics
            psnr_value, ssim_value = self.calculate_statistics(
                gray_image, cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
            )

            # Save denoised image if path is provided
            if self.denoised_dir is not None:
                cv2.imwrite(os.path.join(self.denoised_dir, self.noisy_names[idx]), denoised_image)

            if plot:
                # Show original and denoised images
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

            return denoised_image

        for idx, img_group in enumerate(self.noisy_images):
            if isinstance(img_group, list):  # If img_group is a list of images
                processed_group = [process_single_image(image, idx) for image in img_group]
                denoised_images.append(processed_group)
            else:  # If img_group is a single image
                denoised_images.append(process_single_image(img_group, idx))

        return denoised_images


# Uso de la clase Denoising
# noisy_images_path = "../content/qsd1_w3_test"
# database_path = "../content/BBDD_test"

# output_dir = "output"
# denoised_dir = os.path.join(output_dir, "denoised")
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(denoised_dir, exist_ok=True)

# denoising = Denoising(noisy_images_path, database_path, denoised_dir)
# denoising.process_images()

