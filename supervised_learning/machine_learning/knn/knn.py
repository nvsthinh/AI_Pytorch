import torch
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        """
        Initializes the KNN classifier with the specified number of neighbors.
        
        Args:
            k (int): The number of nearest neighbors to consider for classification.
        """
        self.k = k
        
    def fit(self, X_train, y_train):
        """
        Fits the KNN classifier by storing the training data.
        
        Args:
            X_train (torch.Tensor): Training data features.
            y_train (torch.Tensor): Training data labels.
        """
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        """
        Predicts the labels for the given test data using the KNN algorithm.
        
        Args:
            X_test (torch.Tensor): Test data features.
        
        Returns:
            torch.Tensor: Predicted labels for the test data.
        """
        y_pred = []  # List to store the predicted labels
        
        for test_point in X_test:
            # Calculate the Euclidean distance from the test point to all training points
            distances = torch.norm(self.X_train - test_point, dim=1)
            
            # Get the indices of the k nearest neighbors
            k_nearest_indices = distances.argsort()[:self.k]
            
            # Get the labels of the k nearest neighbors
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            # Find the most common label among the k nearest neighbors
            most_common = Counter(k_nearest_labels.tolist()).most_common(1)
            
            # Append the most common label to the predictions list
            y_pred.append(most_common[0][0])
        
        # Convert the predictions list to a torch tensor and return it
        return torch.tensor(y_pred)