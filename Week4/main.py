import os
import cv2
import pickle

from data_loader import DataLoader
from denoising import Denoising
from painting_detector import PaintingDetector
from descriptors import Descriptors
from evaluator import Evaluator

def compute_bg_removal(imgs_with_bg, imgs_names, output_dir):
    all_cropped={}
    for idx, img in enumerate(imgs_with_bg):
        all_cropped[idx]=[]
    for idx, img in enumerate(imgs_with_bg):
        detector = PaintingDetector(img)
        # Detect and crop paintings
        output_mask, cropped_paintings = detector.detect_and_crop_paintings()
        for i in cropped_paintings:
            all_cropped[idx].append(i)
        # Save mask
        name, ext = imgs_names[idx].rsplit(".", 1)
        filename = f"{name}.png"
        cv2.imwrite(os.path.join(output_dir, filename), output_mask)
    return all_cropped

def main():
    templates_path = "../content/BBDD"
    template_images, template_names = DataLoader({"dataset": templates_path}).load_images_from_folder(extension="jpg", return_names=True)

    if not template_images:
        raise ValueError("Reference images not found. Please check the path.")

    query_path = "../content/qsd1_w4"
    query_images, query_names = DataLoader({"dataset": query_path}).load_images_from_folder(extension="jpg", return_names=True)

    if not query_images:
        raise ValueError("Query images not found. Please check the path.")

    # Create output dir
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Denoise
    denoised_dir = os.path.join(output_dir, "denoised")
    os.makedirs(denoised_dir, exist_ok=True)
    print("Denoising...")
    denoised_images = Denoising(query_images, template_images, noisy_names=query_names, denoised_dir=denoised_dir).process_images(plot=False)
    denoised_images, denoised_names = DataLoader({"dataset": denoised_dir}).load_images_from_folder(extension="jpg", return_names=True)

    # Remove bg
    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)
    print("Removing bg...")
    cropped_paintings = compute_bg_removal(denoised_images, query_names, cropped_dir)
    # Dump the list into a pickle file
    filepath = os.path.join(cropped_dir, 'cropped_paintings.pkl')
    with open(filepath, 'wb') as file:
        pickle.dump(cropped_paintings, file)
    
    cropped_dir = os.path.join(output_dir, "cropped")
    filepath = os.path.join(cropped_dir, 'cropped_paintings.pkl')
    with open(filepath, 'rb') as file:
        cropped_paintings = pickle.load(file)
    
    query_groups=[]
    for idx in cropped_paintings:
        query_groups.append(cropped_paintings[idx])

    kd = Descriptors(template_images, query_groups)
    params = kd.get_default_params('orb')
    (kp_r, des_r, kp_q, des_q) = kd.compute_descriptors('orb', params)
    print(f"kp_r : {len(kp_r)}")
    print(f"des_r : {len(des_r)}")
    print(f"kp_q : {len(kp_q)}")
    print(f"des_q : {len(des_q)}")

    k = 1
    predictions = kd.compute_matches(k_best_results=k, norm=params['norm'], threshold=params['threshold'], min_matches_threshold=params['min_matches_threshold'])

    # Load ground-truth
    gt_path = os.path.join(query_path, 'gt_corresps.pkl')
    with open(gt_path, 'rb') as file:
        gt = pickle.load(file)

    print(f"\nlen predictions: {len(predictions)}")
    print(predictions)
    print(f"\nlen gt: {len(gt)}")
    print(gt)

    # Evaluate
    mapk = Evaluator().mapk(gt, predictions, k)
    print(f"mapk: {mapk}")

if __name__ == '__main__':
    main()