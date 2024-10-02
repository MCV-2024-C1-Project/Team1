class RetrievalSystem:
    def __init__(self, feature_extractor, similarity_calculator):
        self.feature_extractor = feature_extractor
        self.similarity_calculator = similarity_calculator

    def retrieve_top_k(self, query_descriptor, museum_descriptors, k=5):
        # Implementation for retrieving top K results
        return