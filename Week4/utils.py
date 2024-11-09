import cv2
import numpy as np
import matplotlib.pyplot as plt

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