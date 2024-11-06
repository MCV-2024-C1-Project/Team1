import os
import cv2
import pickle
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data_loader import DataLoader
from denoising import Denoising
from painting_detector import PaintingDetector
from retrieval_system import RetrievalSystem
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

def color_sift(image, edgeThreshold=5, nfeatures=200):
    sift = cv2.SIFT_create(edgeThreshold=edgeThreshold, nfeatures=nfeatures)
    keypoints_all = []
    descriptors_all = []
    for i in range(3):
        channel = image[:, :, i]
        keypoints, descriptors = sift.detectAndCompute(channel, None)
        keypoints_all.append(keypoints)
        descriptors_all.append(descriptors)

    keypoints = [kp for sublist in keypoints_all for kp in sublist]
    descriptors = np.vstack(descriptors_all)

    return keypoints, descriptors

def sift_keypoint_detection(image, edgeThreshold=5, nfeatures=300, color_scale='gray'):
    if color_scale=='gray':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(edgeThreshold=edgeThreshold, nfeatures=nfeatures)
        kp, des = sift.detectAndCompute(image, None)
    
    elif color_scale=='rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kp, des = color_sift(image, edgeThreshold=edgeThreshold, nfeatures=nfeatures)

    else:
        raise ValueError(f"Unsupported color scale: '{color_scale}'. Use 'gray' or 'rgb'.")

    return kp, des

def orb_keypoint_detection(image, color_scale='gray'):
    if color_scale=='gray':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(image, None)
    return kp, des

def apply_homography(src, dst, img_q, img_r_best, plot=False):
    m, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)  # m es la matriz que me lleva de la img a identificar al template

    # Detectar producto con la homografía
    orig = np.array([(0,0, 1), (img_r_best.shape[1], 0, 1), (0, img_r_best.shape[0], 1), (img_r_best.shape[1], img_r_best.shape[0], 1)])

    m_inv = np.linalg.inv(m)

    p0 = np.dot(m_inv, orig[0])
    p0 = p0/p0[2]
    punto0 = (int(p0[0]), int(p0[1]))

    p1 = np.dot(m_inv, orig[1])
    p1 = p1/p1[2]
    punto1 = (int(p1[0]), int(p1[1]))

    p2 = np.dot(m_inv, orig[2])
    p2 = p2/p2[2]
    punto2 = (int(p2[0]), int(p2[1]))

    p3 = np.dot(m_inv, orig[3])
    p3 = p3/p3[2]
    punto3 = (int(p3[0]), int(p3[1]))

    obj_det = img_q.copy()
    obj_det = cv2.line(obj_det, punto0, punto1, (0,255,0), 4)
    obj_det = cv2.line(obj_det, punto1, punto3, (0,255,0), 4)
    obj_det = cv2.line(obj_det, punto3, punto2, (0,255,0), 4)
    obj_det = cv2.line(obj_det, punto2, punto0, (0,255,0), 4)

    # Apply homography
    corrected_img = cv2.warpPerspective(img_q, m, (img_r_best.shape[1], img_r_best.shape[0]))

    if plot:
        # Plot
        plt.figure(figsize=(10,8))
        plt.subplot(131)
        plt.imshow(img_r_best)
        plt.title('Template')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(obj_det)
        plt.title('Painting detected')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(corrected_img)
        plt.title('Homography of query image')
        plt.axis('off')

        plt.show(block=False)

    return corrected_img

def compute_homography(img_ref, kp_r, img_query, kp_q, matches):
    dst = np.float32([kp_r[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) # Image template
    src = np.float32([kp_q[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)  # Image query
    corrected_img = apply_homography(src, dst, img_query, img_ref, plot=False)
    return corrected_img

def get_valid_methods_dict():
    ORB = {'threshold': 50, 'norm': cv2.NORM_HAMMING, 'compute_keypoints_and_descriptors': orb_keypoint_detection, 'color_scale': 'gray', 'min_matches_threshold': 32}
    SIFT = {'threshold': 180, 'norm': cv2.NORM_L2, 'compute_keypoints_and_descriptors': sift_keypoint_detection, 'color_scale': 'gray', 'min_matches_threshold': 32}
    COLOR_SIFT = {'threshold': 180, 'norm': cv2.NORM_L2, 'compute_keypoints_and_descriptors': sift_keypoint_detection, 'color_scale': 'rgb', 'min_matches_threshold': 32}
    
    return {'orb': ORB, 'sift': SIFT, 'color_sift': COLOR_SIFT}

def get_matches(descriptors_ref, descriptors_query, norm):
    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(descriptors_ref, descriptors_query)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def plot_matches(img_ref, kp_r, img_query, kp_q, matches, threshold):
    good_matches = [[m] for m in matches if m.distance < threshold]
    img_matches  = cv2.drawMatchesKnn(img_ref, kp_r, img_query, kp_q, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10,10))
    plt.imshow(img_matches)
    plt.title('Good Matches found with ORB')
    plt.axis('off')
    plt.show(block=False)

def print_correspondences(imgs_ref_names, idx_best_match):
    painting = 'The painting is: '
    for l in range(len(imgs_ref_names[idx_best_match])):
        nuevo_caracter = imgs_ref_names[idx_best_match][l]
        if (nuevo_caracter == '_'):
            painting = painting + ' '
        elif (nuevo_caracter == '.'):
            break
        else:
            painting = painting + nuevo_caracter
    print(painting)

def get_predictions(imgs_query, imgs_ref, k_best_results=1, method='orb', imgs_query_names=None, imgs_ref_names=None, output_dir=None):
    # Validate method
    methods = get_valid_methods_dict()
    method_config = methods.get(method)
    if not method_config:
        valid_methods = ', '.join(methods.keys())
        raise ValueError(f"Method not recognized. Use {valid_methods}.")

    results = []
    all_results = []

    # Iterar sobre cada elemento de imgs_query, que puede ser una imagen o una lista de imágenes
    for idx_group, img_group in enumerate(imgs_query):
        print(f"Query group {idx_group+1}/{len(imgs_query)}")
        group_results = []
        
        # Asegurarse de que img_group es una lista (puede contener una o más imágenes)
        if not isinstance(img_group, list):
           img_group = [img_group]
        print("index of img group",idx_group,"shape of img group",len(img_group))
        for idx_q, img_q in enumerate(img_group):
            # Convertir la imagen a RGB
            img_q = cv2.cvtColor(img_q, cv2.COLOR_BGR2RGB)

            # Compute query keypoints and descriptors
            kp_q, des_q = method_config['compute_keypoints_and_descriptors'](img_q, color_scale=method_config['color_scale'])
            if kp_q is None or des_q is None:
                print(f"WARNING: detecting keypoints or descriptors in the query image {idx_q} of group {idx_group}.")
                group_results.append(None)
                continue

            keypoints_r, descriptors_r = ([], [])
            best_matches, n_best_matches = ([], [])
            for idx_r, img_r in enumerate(imgs_ref):
                # Convertir la imagen de referencia a RGB
                img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

                # Compute reference keypoints and descriptors
                kp_r, des_r = method_config['compute_keypoints_and_descriptors'](img_r, color_scale=method_config['color_scale'])
                if kp_r is None or des_r is None:
                    print(f"WARNING: detecting keypoints or descriptors in the reference image {idx_r}.")
                    # Append placeholders to maintain consistent length
                    keypoints_r.append(None)
                    descriptors_r.append(None)
                    best_matches.append([])
                    n_best_matches.append(0)
                    continue
                keypoints_r.append(kp_r)
                descriptors_r.append(des_r)

                # Get matches
                matches = get_matches(des_r, des_q, method_config['norm'])

                # Filter best matches
                good_matches = [m for m in matches if m.distance < method_config['threshold']]
                best_matches.append(good_matches)
                n_best_matches.append(len(good_matches))

            if not n_best_matches:
                print(f"No matches found for the query image {idx_q} in group {idx_group}.")
                group_results.append(None)
                continue

            # Guardar los resultados de las mejores coincidencias
            group_results.append(n_best_matches)

            idx_best_match = np.argmax(n_best_matches)
            kp_r_best = keypoints_r[idx_best_match]
            des_r_best = descriptors_r[idx_best_match]
            img_r_best = cv2.cvtColor(imgs_ref[idx_best_match], cv2.COLOR_BGR2RGB)

            matches = get_matches(des_r_best, des_q, method_config['norm'])

            # Plot matches (opcional)
            # plot_matches(img_r_best, kp_r_best, img_q, kp_q, matches, method_config['threshold'])

            # # Compute homography using best match (opcional)
            # corrected_img = compute_homography(img_r_best, kp_r_best, img_q, kp_q, matches)

            # # Print correspondences (opcional)
            # if imgs_ref_names is not None:
            #     print_correspondences(imgs_ref_names, idx_best_match)
            #     if output_dir is not None:
            #         output_path = os.path.join(output_dir, "corrected_imgs")
            #         os.makedirs(output_path, exist_ok=True)
            #         plt.imsave(os.path.join(output_path, imgs_ref_names[idx_best_match]), corrected_img)

        # Añadir los resultados del grupo al resultado final
        results.append(group_results)
    results = [group[0] if len(group) == 1 else group for group in results]

    top_results = []
    for group in results:
        print(len(group))
        top_k = []
        for result in group:
            if not isinstance(result, list):
                print(f'max value: {np.max(np.array(group))}')
                if np.max(np.array(group)) < method_config['min_matches_threshold']:
                    top_k.append([-1])  # Vector de -1s de longitud k
                    break
                top_k.append(RetrievalSystem().retrieve_top_k(np.array(group).reshape(1, -1), reverse=True, k=k_best_results))
                break
            print(f'max value: {np.max(np.array(result))}')
            if np.max(np.array(result)) < method_config['min_matches_threshold']:
                top_k.append([-1])  # Vector de -1s de longitud k
            else:
                top_k.append(RetrievalSystem().retrieve_top_k(np.array(result).reshape(1, -1), reverse=True, k=k_best_results))
        top_results.append(top_k)
    
    top_results = [group[0] if len(group) == 1 else group for group in top_results]
    predictions = []
    for item in top_results:
        if isinstance(item, list):
            # Si el primer elemento es una lista y contiene otro nivel de listas, aplana
            flat_item = [subitem for sublist in item for subitem in sublist] if isinstance(item[0], list) else item
            predictions.append(flat_item)
        else:
            predictions.append(item)

    return predictions


def main():
    templates_path = "C:/Users/laila/Downloads/BBDD/BBDD"
    template_images, template_names = DataLoader({"dataset": templates_path}).load_images_from_folder(extension="jpg", return_names=True)
    
    query_path = "C:/Users/laila/Downloads/qsd1_w4/qsd1_w4"
    query_images, query_names = DataLoader({"dataset": query_path}).load_images_from_folder(extension="jpg", return_names=True)

    # Create output dir
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
        
    # Denoise
    denoised_dir = os.path.join(output_dir, "denoised")
    os.makedirs(denoised_dir, exist_ok=True)
    print("Denoising...")
    denoised_images = Denoising(query_images, template_images, noisy_names=query_names, denoised_dir=denoised_dir).process_images(plot=False)
    denoised_images, denoised_names = DataLoader({"dataset": denoised_dir}).load_images_from_folder(extension="jpg", return_names=True)
    # Load images with noise and bg
    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)
    print("Removing bg...")
    cropped_paintings = compute_bg_removal(denoised_images, query_names, cropped_dir)
    # # Dump the list into a pickle file
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
        
    # Get predictions
    k_best_results = 1
    predictions = get_predictions(query_groups, template_images, k_best_results=k_best_results, method='orb', imgs_ref_names=template_names, output_dir=output_dir)

    # Load ground-truth
    gt_path = os.path.join(query_path, 'gt_corresps.pkl')
    with open(gt_path, 'rb') as file:
        gt = pickle.load(file)

    print(f"\nlen predictions: {len(predictions)}")
    print(predictions)
    print(f"\nlen gt: {len(gt)}")
    print(gt)

    # Evaluate
    mapk = Evaluator().mapk(gt, predictions, k_best_results)
    print(f"mapk: {mapk}")

    input("Press enter to exit")

if __name__ == '__main__':
    main()