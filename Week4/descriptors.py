import cv2
import numpy as np
import matplotlib.pyplot as plt

from retrieval_system import RetrievalSystem

class Descriptors():
    def __init__(self, db_ref, db_query):
        if isinstance(db_ref, list):
            self.db_ref = db_ref
        else:
            raise TypeError("Input db_ref must be a list of images.")
        
        if isinstance(db_query, list):
            self.db_query = db_query
        else:
            raise TypeError("Input db_query must be a list of images.")
        
        self.kp_r, self.des_r = (None, None)
        self.kp_q, self.des_q = (None, None)
        
        self.kp_r_best, self.des_r_best, self.img_r_best = ([], [], [])
        self.no_match_value = -1
        

    def select_db(self, db):
        if db == 'ref':
            return self.db_ref
        if db == 'query':
            return self.db_query
        raise ValueError("db name not valid. Must be 'ref' for reference db or 'query' for query db.")

    def get_valid_methods_dict(self):
        return {'orb': self.compute_orb_descriptor, 'sift': self.compute_sift_descriptor, 'color_sift': self.compute_color_sift_descriptor}
    
    def get_default_params(self, method):
        if method == 'orb':
            params = {'threshold': 50, 'norm': cv2.NORM_HAMMING, 'min_matches_threshold': 32}
        elif method == 'sift':
            params = {'threshold': 180, 'norm': cv2.NORM_L2, 'min_matches_threshold': 28, 'edgeThreshold': 5, 'nfeatures': 300}
        elif method == 'color_sift':
            params = {'threshold': 180, 'norm': cv2.NORM_L2, 'min_matches_threshold': 35, 'edgeThreshold': 5, 'nfeatures': 200}
        else:
            raise ValueError(f"Method not valid. Available methods: {self.get_valid_methods_dict.keys()}")
        return params

    def compute_descriptors(self, method, params):
        valid_methods = self.get_valid_methods_dict()
        if method not in valid_methods.keys():
            raise ValueError(f"Method not valid. Available methods: {valid_methods.keys()}")
        # Compute descriptors
        imgs = self.select_db('ref')
        self.kp_r, self.des_r = valid_methods[method](imgs, **params)
        imgs = self.select_db('query')
        self.kp_q, self.des_q = valid_methods[method](imgs, **params)
        return (self.kp_r, self.des_r, self.kp_q, self.des_q)
        
    def compute_orb_descriptor(self, imgs, threshold=50, norm=cv2.NORM_HAMMING, min_matches_threshold=32):
        keypoints, descriptors = ([], [])
        for idx_group, img_group in enumerate(imgs):
            if not isinstance(img_group, list):
                img_group = [img_group]
            print(f"Group {idx_group+1}/{len(imgs)}, group size: {len(img_group)}", end='\r')
            keypoints_group, descriptors_group = ([], [])
            for idx, img in enumerate(img_group):
                kp, des = self.orb_keypoint_detection(img, threshold=threshold, norm=norm, min_matches_threshold=min_matches_threshold)
                if kp is None or des is None:
                    print(f"Warning: non keypoints or descriptors detected in image {idx} of group {idx_group}.")
                    keypoints_group.append(None)
                    descriptors_group.append(None)
                    continue
                keypoints_group.append(kp)
                descriptors_group.append(des)
            keypoints.append(keypoints_group)
            descriptors.append(descriptors_group)
        print('\n')
        return keypoints, descriptors
            
    def compute_sift_descriptor(self, imgs, threshold=180, norm=cv2.NORM_L2, min_matches_threshold=32, edgeThreshold=5, nfeatures=200):
        keypoints, descriptors = ([], [])
        for idx_group, img_group in enumerate(imgs):
            if not isinstance(img_group, list):
                img_group = [img_group]
            print(f"Group {idx_group+1}/{len(imgs)}, group size: {len(img_group)}", end='\r')
            keypoints_group, descriptors_group = ([], [])
            for idx, img in enumerate(img_group):
                kp, des = self.sift_keypoint_detection(img, threshold=threshold, norm=norm, min_matches_threshold=min_matches_threshold, edgeThreshold=edgeThreshold, nfeatures=nfeatures)
                if kp is None or des is None:
                    print(f"Warning: non keypoints or descriptors detected in image {idx} of group {idx_group}.")
                    keypoints_group.append(None)
                    descriptors_group.append(None)
                    continue
                keypoints_group.append(kp)
                descriptors_group.append(des)
            keypoints.append(keypoints_group)
            descriptors.append(descriptors_group)
        print('\n')
        return keypoints, descriptors
    
    def compute_color_sift_descriptor(self, imgs, threshold=180, norm=cv2.NORM_L2, min_matches_threshold=32, edgeThreshold=5, nfeatures=200):
        keypoints, descriptors = ([], [])
        for idx_group, img_group in enumerate(imgs):
            if not isinstance(img_group, list):
                img_group = [img_group]
            print(f"Group {idx_group+1}/{len(imgs)}, group size: {len(img_group)}", end='\r')
            keypoints_group, descriptors_group = ([], [])
            for idx, img in enumerate(img_group):
                kp, des = self.color_sift_keypoint_detection(img, threshold=threshold, norm=norm, min_matches_threshold=min_matches_threshold, edgeThreshold=edgeThreshold, nfeatures=nfeatures)
                if (kp is None or des is None) or (None in kp or None in des):
                    print(f"Warning: non keypoints or descriptors detected in image {idx} of group {idx_group}.")
                    keypoints_group.append(None)
                    descriptors_group.append(None)
                    continue
                keypoints_group.append(kp)
                descriptors_group.append(des)
            keypoints.append(keypoints_group)
            descriptors.append(descriptors_group)
        print('\n')
        return keypoints, descriptors

    def orb_keypoint_detection(self, image, threshold=50, norm=cv2.NORM_HAMMING, min_matches_threshold=32):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(image, None)
        return kp, des

    def color_sift_keypoint_detection(self, image, threshold=180, norm=cv2.NORM_L2, min_matches_threshold=32, edgeThreshold=5, nfeatures=200):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    def sift_keypoint_detection(self, image, threshold=180, norm=cv2.NORM_L2, min_matches_threshold=32, edgeThreshold=5, nfeatures=300):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(edgeThreshold=edgeThreshold, nfeatures=nfeatures)
        kp, des = sift.detectAndCompute(image, None)

        return kp, des

    def compute_matches(self, k_best_results, norm, threshold, min_matches_threshold):
        results = []
        for idx_q_group, des_q_group in enumerate(self.des_q):
            group_results = []
            for idx_q, des_q in enumerate(des_q_group):
                if des_q is None:
                    print(f"WARNING: detecting keypoints or descriptors in the query image {idx_q} of group {idx_q_group}.")
                    group_results.append(None)
                    continue
                best_matches, n_best_matches = ([], [])
                for idx_r, des_ref in enumerate(self.des_r):
                    des_r = des_ref[0]
                    if des_r is None:
                        best_matches.append([])
                        n_best_matches.append(0)
                        continue
                    # print(f"(type(des_r), type(des_q)) : {(type(des_r), type(des_q))}")
                    # Get matches
                    matches = self.get_matches(des_r, des_q, norm)
                    # Filter best matches
                    good_matches = [m for m in matches if m.distance < threshold]
                    best_matches.append(good_matches)
                    n_best_matches.append(len(good_matches))
                if not n_best_matches:
                    print(f"Warning: No matches found for the query image {idx_q} in group {idx_group}.")
                    group_results.append(None)
                    continue
                group_results.append(n_best_matches)
                
                # Save best match
                idx_best_match = np.argmax(n_best_matches)
                self.kp_r_best.append(self.kp_r[idx_best_match])
                self.des_r_best.append(self.des_r[idx_best_match])
                self.img_r_best.append(self.db_ref[idx_best_match])
            
            results.append(group_results)
        results = [group[0] if len(group) == 1 else group for group in results]

        top_k = self.get_top_k_results(results, k_best_results=k_best_results, min_matches_threshold=min_matches_threshold)
        predictions = self.get_predictions(top_k)
        return predictions

    def get_predictions(self, top_k):
        predictions = []
        for item in top_k:
            if isinstance(item, list):
                flat_item = [subitem for sublist in item for subitem in sublist] if isinstance(item[0], list) else item
                predictions.append(flat_item)
            else:
                predictions.append(item)
        return predictions

    def is_valid_match(self, result):
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        
        min_value = np.min(result)
        max_value = np.max(result)
        second_max_value = np.max(result[result != max_value])
        p_10 = np.count_nonzero(result < 10) / len(result)
        # print(max_value - second_max_value)
        # print(p_10)
        # if (p_10 > 0.95):
        #     return 1
        return 0


    def get_top_k_results(self, results, k_best_results, min_matches_threshold):
        top_results = []
        for group in results:
            top_k = []
            for result in group:
                if not isinstance(result, list):
                    if np.max(np.array(group)) < min_matches_threshold:
                        if (not self.is_valid_match(group)):
                            top_k.append([self.no_match_value])
                            break
                    top_k.append(RetrievalSystem().retrieve_top_k(np.array(group).reshape(1, -1), reverse=True, k=k_best_results))
                    break
                if np.max(np.array(result)) < min_matches_threshold:
                    if (not self.is_valid_match(result)):
                        top_k.append([self.no_match_value])
                    else:
                        top_k.append(RetrievalSystem().retrieve_top_k(np.array(result).reshape(1, -1), reverse=True, k=k_best_results))
                else:
                    top_k.append(RetrievalSystem().retrieve_top_k(np.array(result).reshape(1, -1), reverse=True, k=k_best_results))
            top_results.append(top_k)
        top_results = [group[0] if len(group) == 1 else group for group in top_results]
        return top_results

    def get_matches(self, descriptors_ref, descriptors_query, norm):
        bf = cv2.BFMatcher(norm, crossCheck=True)
        matches = bf.match(descriptors_ref, descriptors_query)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def plot_matches(self, img_ref, kp_r, img_query, kp_q, matches, threshold):
        good_matches = [[m] for m in matches if m.distance < threshold]
        img_matches  = cv2.drawMatchesKnn(img_ref, kp_r, img_query, kp_q, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(10,10))
        plt.imshow(img_matches)
        plt.title('Good Matches found with ORB')
        plt.axis('off')
        plt.show(block=False)

    def print_correspondences(self, imgs_ref_names, idx_best_match):
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

    