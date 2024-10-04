class RetrievalSystem:
    def __init__(self, feature_extractor, similarity_calculator):
        self.feature_extractor = feature_extractor
        self.similarity_calculator = similarity_calculator

     def retrieve_top_k(self, query_descriptor, museum_descriptors, k=10):
        """
        Retrieve the top k=10 most similar images from the museum descriptors based on the query descriptor.

        Args:
            query_descriptor: Feature vector of query image
            museum_descriptors: List containing index, filename, color_space
            k(int): Number of most similar matches to return


        Returns:
            list: A list of the top k=10 results where each result is [image_index, similarity_score]
        """
        similarities = []

        for index, _, _, museum_descriptor in museum_descriptors:
            score = self.similarity_calculator.calculate_similarity(query_descriptor, museum_descriptor)
            similarities.append((index, score))

        # Sort similarities in a descending order, based on similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_results = similarities[:k]
    
    return top_k_results
