class RetrievalSystem:

<<<<<<< HEAD
    def search_top_k_values(self,color_scales_list, color_name, method="Correlation", K=1):
        values_with_indices = []
        for i, sublist in enumerate(color_scales_list):
            for value, name in sublist:
                if name == color_name:
                    values_with_indices.append((value, i))
                    break
        
        # Determine if sorting should be reversed
        reverse_sort = method in ["Correlation", "Intersection"]
        
        sorted_values_with_indices = sorted(values_with_indices, key=lambda x: x[0], reverse=reverse_sort)
        
        # Get the top K values
        top_k_values_with_indices = sorted_values_with_indices[:K]
        top_k_indices = [index for _, index in top_k_values_with_indices]
        return top_k_indices

=======
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
>>>>>>> 6daedb03a16dfd9ac2935aff96ca8fe4cc97825c
