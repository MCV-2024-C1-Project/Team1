class RetrievalSystem:

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

