import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from FaceDataset import FaceDataset
from PCA import PCA
from FaceRecognizer import FaceRecognizer

IMAGE_SIZE = (90, 90)

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    IMAGE_SIZE = (90,90)

    # Load dataset
    dataset = FaceDataset(path='data/orl_faces', image_size=IMAGE_SIZE, grayscale=False)
    X, y, label_map = dataset.load_images()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

  


