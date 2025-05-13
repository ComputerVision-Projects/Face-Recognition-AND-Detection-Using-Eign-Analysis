import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class FaceRecognizer:
    def __init__(self, pca, n_neighbors=1):
        self.pca = pca
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.X_train_pca = None
        self.y_train = None
        self.X_train = None  # Store original images
    
    def train(self, X_train, y_train):
        """Train the face recognizer
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Original images (not PCA transformed)
        y_train : array-like, shape (n_samples,)
            Labels for the training images
        """
        self.X_train = X_train
        self.y_train = y_train
        # Apply PCA to the original images
        self.X_train_pca = self.pca.transform(X_train)
        # Train the classifier on the PCA-transformed data
        self.classifier.fit(self.X_train_pca, y_train)
    
    def predict(self, X_test):
        """Predict labels for test images
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test images (not PCA transformed)
        """
        X_test_pca = self.pca.transform(X_test)
        return self.classifier.predict(X_test_pca)
    
   