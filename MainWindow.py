from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QPushButton, QLabel, QSlider
import os
import sys
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
from PIL import Image
import cv2
import numpy as np
from ImageViewer import ImageViewer
import joblib
from FaceDataset import FaceDataset
from PCA import PCA
from FaceRecognizer import FaceRecognizer


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        loadUi("MainWindow.ui", self)

        #corner detector widgets
        self.input_image = self.findChild(QWidget, "input")
        self.output_image1 = self.findChild(QWidget, "output1")
        self.output_image2 = self.findChild(QWidget, "output2")
        self.output_image2 = self.findChild(QWidget, "output2")
        self.output_image3 = self.findChild(QWidget, "output3")


        self.input_viewer = ImageViewer(input_view=self.input_image, mode=False)
        self.output_viewer1 = ImageViewer(output_view=self.output_image1, mode=False)
        self.output_viewer2 = ImageViewer(output_view=self.output_image2, mode=False)
        self.output_viewer3 = ImageViewer(output_view=self.output_image3, mode=False)

        #corner detector parameters
        self.components_num = self.findChild(QSlider, "componentSlider")

        self.components_slider_label = self.findChild(QLabel, "componentLabel")

        self.components_num.valueChanged.connect(self.update_component_label)

        #corner detector methods
        self.apply_recognition = self.findChild(QPushButton, "applyButton")
        self.apply_recognition.clicked.connect(self.apply)

        # Paths
        #self.dataset_path = "data/orl_faces"
        self.image_shape = (100,100)

        #Main Window
        #  Load pre-trained models
        self.pca = None
        self.knn = None
        self.label_map=None
        self.X_train =None
        self.y_train =None
        self.load_trained_models()

    def load_trained_models(self):
        self.pca = joblib.load("models/pca1.pkl")
        self.knn = joblib.load("models/knn1.pkl")
        #self.label_map = joblib.load("label_map.pkl")
        self.X_train = joblib.load("models/train_images1.pkl")  # Original images
        self.y_train = joblib.load("models/train_labels1.pkl")
        # Reverse label map: {0: 's1', 1: 's2', ...}
        #self.label_map_rev = {v: k for k, v in self.label_map.items()}
        
        # Optional: precompute PCA transforms of training data
        #self.X_train_pca = self.pca.transform(self.X_train)

    # def get_most_similar(self, input_image):
    #     """Find the most similar image in the training set
        
    #     Parameters:
    #     -----------
    #     input_image : array-like, shape (n_features,)
    #         A single input image (not PCA transformed)
    #     """
    #     # Reshape input_image to match expected shape if needed
    #     if input_image.ndim == 1:
    #         input_image = input_image.reshape(1, -1)
            
    #     # Transform the input image using PCA
        
    #     # Calculate Euclidean distances to all training samples
    #     # dists = []
    #     # for i in range(len(self.X_train_pca)):
    #     #     dist = np.linalg.norm(self.X_train_pca[i] - input_pca[0])
    #     #     dists.append(dist)
        
    #     # Find the index of the minimum distance
    #     # min_idx = np.argmin(dists)
    #     output= self.knn.predict(input_image)

    #     print(output)

    #     #return self.X_train[min_idx], self.y_train[min_idx]
    
    def apply(self):
        image = self.input_viewer.get_loaded_image()
        if image is None:
            print("No Image Loaded.")
            return
        
        # Print the shape for debugging
        print(f"Input image shape: {image.shape}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
       
        # Resize to match the expected image size
        image_resized = cv2.resize(image_gray, self.image_shape)
        image_flattened = image_resized.flatten().reshape(1, -1)
        img_pca = self.pca.transform(image_flattened)
        img_pca = np.real(img_pca)
        label = self.knn.predict(img_pca)
        neighbors = self.knn.kneighbors(img_pca, return_distance=False)[0]
        for i, idx in enumerate(neighbors):
            neighbor_img = np.array(self.X_train[idx]).reshape(100, 100)
            if i==0:
                self.output_viewer1.display_output_image(neighbor_img)
            if i==1:
                 self.output_viewer2.display_output_image(neighbor_img)
            if i==2:
                 self.output_viewer3.display_output_image(neighbor_img)


   
        #match_image, match_label = self.get_most_similar(image_flattened)
        
        # Display the matching image
        print(len(neighbors))
        print(label)
        
       


    def update_component_label(self):
        component_label = self.components_num.value()
        self.components_slider_label.setText(str(component_label))    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.showMaximized()
    sys.exit(app.exec_())        
    

