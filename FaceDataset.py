import os
import numpy as np
import re
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

class FaceDataset:
    def __init__(self, path, image_size, grayscale=True):
        self.path = path
        self.image_size = image_size
        self.grayscale = grayscale

    def load_images(self):
        '''
        Assigns a numeric label to each person, resizes and  flattens images, and returns data ready for training/testing.
        - X: stores **flattened image arrays**
        - y: stores **corresponding numeric labels**
        - label_map: dictionary mapping each folder name (e.g. `s1`) to an integer (e.g. `0`)
        
        This version ensures that folders are properly sorted numerically:
        's1' -> 0, 's2' -> 1, 's3' -> 2, etc.
        '''
        X, y = [], []
        
        # Get all person directories and sort them correctly using numeric value
        all_persons = []
        for person in os.listdir(self.path):
            person_dir = os.path.join(self.path, person)
            if os.path.isdir(person_dir):
                all_persons.append(person)
        
        # Extract numeric part and sort based on that
        def get_numeric_value(folder_name):
            # Extract numeric part from folder name (e.g., 's1' -> 1, 's10' -> 10)
            match = re.search(r'(\d+)', folder_name)
            if match:
                return int(match.group(1))
            return 0  # Default value if no number found
            
        # Sort folders by their numeric values
        all_persons.sort(key=get_numeric_value)
        
        # Create label map with correct order
        label_map = {person: idx for idx, person in enumerate(all_persons)}
        
        print("Label mapping created:")
        for person, label in sorted(label_map.items(), key=lambda x: x[1]):
            print(f"  {person}: {label}")
            
        # Now process all images with the correct label mapping
        for person in all_persons:
            person_dir = os.path.join(self.path, person)
            person_label = label_map[person]
            
            # Process each image
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                
                try:
                    img = imread(img_path)
                    
                    # Handle grayscale conversion if needed
                    if self.grayscale and len(img.shape) == 3:
                        img = rgb2gray(img)
                    
                    # Resize and flatten
                    img = resize(img, self.image_size).flatten()
                    
                    # Add to dataset
                    X.append(img)
                    y.append(person_label)
                    
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"Dataset loaded: {X.shape[0]} images, {len(label_map)} unique people")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(y)
        print(label_map)
        
        return X, y, label_map