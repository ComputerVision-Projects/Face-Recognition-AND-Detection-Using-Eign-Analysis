import os
import joblib
from sklearn.model_selection import train_test_split
from FaceDataset import FaceDataset
from PCA import PCA
from FaceRecognizer import FaceRecognizer
import cv2

# Parameters
image_shape = (90, 90)
dataset_path = "data/orl_faces"
n_components = 1000  # Reduced from 1000, should be less than min(n_samples, n_features)
k_neighbors = 2

# Load dataset (grayscale is typically preferred)
dataset = FaceDataset(dataset_path, image_size=image_shape, grayscale=False)
X, y, label_map = dataset.load_images()

print(f"Loaded data: X shape={X.shape}, y shape={y.shape}")

# Apply PCA
pca = PCA(n_components=n_components)
pca.fit(X)  # Just fit the PCA, don't transform yet

# Create and train recognizer with original images
recognizer = FaceRecognizer(pca=pca, n_neighbors=k_neighbors)
recognizer.train(X, y)  # Pass original images, PCA is applied inside

# Get the PCA-transformed images for saving
X_pca = pca.transform(X)
# Save components
joblib.dump(pca, "trained_pca.pkl")
joblib.dump(recognizer.classifier, "trained_knn.pkl")
joblib.dump(label_map, "label_map.pkl")
joblib.dump(X, "X_train.pkl")          # Original images
joblib.dump(y, "y_train.pkl")          # Labels
joblib.dump(X_pca, "X_train_pca.pkl")  # PCA-transformed images, useful for comparison

print("Model trained and saved!")