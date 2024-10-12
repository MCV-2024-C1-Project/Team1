import cv2
import numpy as np
import matplotlib.pyplot as plt
import textwrap

from data_loader import DataLoader

def convert_image(img, color_scale='HSV', plot=False):
    """_summary_

    Args:
        img (numpy.ndarray): image to convert. Must be BGR.
        color_scale (str, optional): color scale to which convert img. Defaults to 'HSV'.
        plot (bool, optional): if True, img is plotted in color_scale, together with each channel. Defaults to False.

    Raises:
        ValueError: If color scale is not valid.

    Returns:
        numpy.ndarray: converted image
    """
    color_scale_dict = {
        'Gray'  : cv2.COLOR_BGR2GRAY,
        'HSV'   : cv2.COLOR_BGR2HSV,
        'YCrCb' : cv2.COLOR_BGR2YCrCb,
        'Lab'   : cv2.COLOR_BGR2Lab,
    }
    if (color_scale not in color_scale_dict.keys()):
        raise ValueError(f"Valid color spaces: {color_scale_dict.keys()}")
    
    converted_image = cv2.cvtColor(img, color_scale_dict[color_scale])
    channel_0, channel_1, channel_2 = cv2.split(converted_image)

    if plot:
        plt.figure()

        plt.subplot(1,4,1)
        plt.imshow(img)
        plt.title(f"{color_scale} image")
        plt.axis('off')
        
        plt.subplot(1,4,2)
        plt.imshow(channel_0)
        plt.title(f"{color_scale[0]} channel")
        plt.axis('off')

        plt.subplot(1,4,3)
        plt.imshow(channel_1)
        plt.title(f"{color_scale[1]} channel")
        plt.axis('off')

        plt.subplot(1,4,4)
        plt.imshow(channel_2)
        plt.title(f"{color_scale[2]} channel")
        plt.axis('off')
        plt.show(block=False)
    
    return converted_image

# Background sampling from top, bottom, left, and right borders for S and V channels
def extract_borders(channel, n_pix_bg=10):
    """Extracts pixel values from the top, bottom, left, and right borders of a given image channel.

    Args:
        channel (numpy.ndarray): A 2D array representing a single channel of an image.

    Returns:
        numpy.ndarray: A 1D array containing the concatenated pixel values from the top, bottom, left, and right borders of the input channel.
    """
    top = channel[:n_pix_bg, :]
    bottom = channel[-n_pix_bg:, :]
    left = channel[:, :n_pix_bg]
    right = channel[:, -n_pix_bg:]
    return np.concatenate((top.flatten(), bottom.flatten(), left.flatten(), right.flatten()))

def calc_bg_statistics(channel, n_pix_bg=10):
    """Get background statistics

    Args:
        channel (numpy.ndarray): Array corresponding to one channel of an image.
        n_pix_bg (int, optional): Number of pixels to sample from each border of the image for background extraction. Defaults to 10.

    Returns:
        tuple: A tuple containing two values:
                - mean (float): The mean of the background pixel values.
                - std (float): The standard deviation of the background pixel values.
    """
    bg_pixels = extract_borders(channel, n_pix_bg)
    return (np.mean(bg_pixels), np.std(bg_pixels))

def get_channel_mask(channel, n_pix_bg=10, threshold=2.5):
    mean, std = calc_bg_statistics(channel=channel, n_pix_bg=n_pix_bg)
    print(f"Saturation - Mean: {mean}, Std Dev: {std}")

    mask = np.abs(channel - mean) > (threshold * std)
    percentage = (np.sum(mask)/channel.size)
    print(f"Number of True elements in mask: {percentage*100}%")

    return mask, percentage

def delete_exterior_elements(mask):
    # Identificar la región más grande y crear una máscara que la contenga
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    largest_component_index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Ignorar el fondo (índice 0)
    
    idx_i, idx_j = np.where(labels == largest_component_index)
    min_i, max_i = np.min(idx_i), np.max(idx_i)
    min_j, max_j = np.min(idx_j), np.max(idx_j)
    vertices = np.array([[(min_j, min_i), (max_j, min_i), (max_j, max_i), (min_j, max_i)]], dtype=np.int32)
    mask_aux = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(mask_aux, vertices, 255)

    mask = cv2.bitwise_and(mask.astype(np.uint8), mask.astype(np.uint8), mask=mask_aux)

    return mask

def apply_morph_operations(binary_img, kernel_size=(5, 5), n_iter=1, connectivity=4):
    kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv2.erode(binary_img, kernel, iterations=n_iter)

    mask_aux = delete_exterior_elements(eroded_image)
    result = cv2.bitwise_and(eroded_image, eroded_image, mask=mask_aux)
    
    dilated_image = cv2.dilate(result, kernel, iterations=n_iter)
    dilated_image = dilated_image / 255

    return dilated_image

def get_combined_mask(ch_a_info, ch_b_info, kernel_size=(5,5), n_iter=1, connectivity=4):
    mask_ch_a, percentage_a = ch_a_info
    mask_ch_b, percentage_b = ch_b_info

    cv2.imwrite('mask_ch_a.png', mask_ch_a.astype(np.uint8)*255)
    cv2.imwrite('mask_ch_b.png', mask_ch_b.astype(np.uint8)*255)
    
    if (percentage_a < 0.13 and percentage_b > 0.45):
        # Use only channel b
        combined_mask = apply_morph_operations(mask_ch_b.astype(np.uint8) * 255, kernel_size=kernel_size, n_iter=n_iter, connectivity=connectivity)
    elif (percentage_b < 0.13 and percentage_a > 0.45):
        # Use only channel a
        combined_mask = apply_morph_operations(mask_ch_a.astype(np.uint8) * 255, kernel_size=kernel_size, n_iter=n_iter, connectivity=connectivity)
    else:
        # Use both channels
        if (percentage_a < 0.30 and percentage_b < 0.30):
            # Do not apply morphology
            combined_mask = np.logical_or(mask_ch_a, mask_ch_b).astype(np.uint8)
        else:
            if (percentage_a < 0.10):
                # Apply morphology only to channel b and then combine channels
                mask_ch_b = apply_morph_operations(mask_ch_b.astype(np.uint8)*255, kernel_size=kernel_size, n_iter=n_iter, connectivity=connectivity)
                combined_mask = np.logical_or(mask_ch_a, mask_ch_b).astype(np.uint8)
            elif (percentage_b < 0.10):
                # Apply morphology only to channel a and then combine channels
                mask_ch_a = apply_morph_operations(mask_ch_a.astype(np.uint8)*255, kernel_size=kernel_size, n_iter=n_iter, connectivity=connectivity)
                combined_mask = np.logical_or(mask_ch_a, mask_ch_b).astype(np.uint8)
            else:
                # Combine channels and then apply morphology
                mask_a_b = np.logical_or(mask_ch_a, mask_ch_b)
                combined_mask = apply_morph_operations(mask_a_b.astype(np.uint8)*255, kernel_size=kernel_size, n_iter=n_iter, connectivity=connectivity)
    
    cv2.imwrite('combined_mask.png', combined_mask.astype(np.uint8) * 255)

    mask = delete_exterior_elements(combined_mask)

    cv2.imwrite('deleted_ext_elements.png', mask.astype(np.uint8) * 255)

    # Apply closure
    kernel = np.ones((90, 90), np.uint8)
    closing = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('final_mask_closed.png', closing.astype(np.uint8) * 255)

    return closing

def get_bg_mask(hsv_image, n_pix_bg=10, threshold=2.5, kernel_size=(5,5), n_iter=1, connectivity=4):
    h_channel, s_channel, v_channel = cv2.split(hsv_image)
    s_mask, s_percentage = get_channel_mask(channel=s_channel, n_pix_bg=n_pix_bg, threshold=threshold)
    v_mask, v_percentage = get_channel_mask(channel=v_channel, n_pix_bg=n_pix_bg, threshold=threshold)

    mask = get_combined_mask((s_mask, s_percentage), (v_mask, v_percentage), kernel_size=(7,7), n_iter=3, connectivity=4)

    return mask

def remove_bg(img, n_pix_bg=10, threshold=2.5, kernel_size=(5,5), n_iter=1, connectivity=4):
    # Convert image to HSV
    hsv_img = convert_image(img, color_scale='HSV', plot=True)
    
    bg_mask = get_bg_mask(hsv_img, n_pix_bg=10, threshold=2.5, kernel_size=(7,7), n_iter=3, connectivity=4)

    final_hsv = cv2.bitwise_and(hsv_img, hsv_img, mask=bg_mask.astype(np.uint8) * 255)
    final_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('hsv_image_with_foreground.png', final_bgr)
    return final_bgr

def main():
    imgs_in_with_bg = DataLoader({"dataset": "../content/qsd2_w2/qsd2_w1_test"}).load_images_from_folder()
    img = imgs_in_with_bg[1]
    img_without_bg = remove_bg(img, kernel_size=(7,7), n_iter=3, connectivity=4)
    cv2.imwrite("img_without_bg.png", img_without_bg)

    # Wait for user action to exit
    input("Press Enter to exit")

if __name__ == "__main__":
    main()