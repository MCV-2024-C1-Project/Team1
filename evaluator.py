class Evaluator:
    def evaluate(self, predictions, ground_truth):
       
        # Count the number of correct predictions
        correct_predictions = sum(p == t for p, t in zip(predictions, ground_truth))
        
        # Calculate the accuracy percentage
        accuracy = (correct_predictions / len(ground_truth)) * 100
        
        return accuracy
        