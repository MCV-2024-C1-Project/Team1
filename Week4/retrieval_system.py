import numpy as np

class RetrievalSystem:
    def retrieve_top_k(self, scores, reverse=False, k=10):
        """
        Retrieve the top k=10 most similar images from the museum descriptors based on the query descriptor.

        Args:
            scores (numpy.ndarray): similarity matrix s[n,m], where n is the number of inputs and m is the number of reference histograms. s[i,j] shows the similarity between input i and reference j.
            k (int): Number of most similar matches to return


        Returns:
            list: A list of the top k=10 results where each result is [image_index, similarity_score]
        """

        if k > scores.shape[1]:
            raise ValueError(f"Invalid number of results '{k}'.\nMust be equal or lower than: {scores.shape[1]}")
        
        if reverse:
            sorted_scores = np.argsort(scores, axis=1)[:,::-1]
        
        top_k_results = sorted_scores[:, :k]

        if len(top_k_results.tolist())==1:
            return top_k_results.tolist()[0]
        else:
            return top_k_results.tolist()
