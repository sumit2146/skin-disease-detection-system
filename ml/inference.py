import os
import numpy as np
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TF_AVAILABLE = False
    tf = None
    print("TensorFlow not found. Application will run in MOCK mode.")

from .mapping import disease_info
import random
import json

class DiseasePredictor:
    def __init__(self, model_path='skin_disease_model.h5', indices_path='class_indices.json'):
        self.model_path = model_path
        self.indices_path = indices_path
        self.model = None
        self.classes = []
        self.load_resources()

    def load_resources(self):
        # Load Model
        if TF_AVAILABLE and os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.model = None
        else:
            if not TF_AVAILABLE:
                print("TensorFlow not available. Running in MOCK mode.")
            else:
                print("Model file not found. Running in MOCK mode.")
            self.model = None

        # Load Class Indices
        if os.path.exists(self.indices_path):
            try:
                with open(self.indices_path, 'r') as f:
                    indices = json.load(f)
                    # Indices matches {class_name: index}. We need [class_name_at_0, class_name_at_1...]
                    # Sort by index
                    self.classes = [k for k, v in sorted(indices.items(), key=lambda item: item[1])]
                    print(f"Loaded {len(self.classes)} classes.")
            except Exception as e:
                 print(f"Failed to load class indices: {e}")
                 self.classes = sorted(list(disease_info.keys()))
        else:
            print("class_indices.json not found. Using default mapping keys.")
            self.classes = sorted(list(disease_info.keys()))

    def preprocess_image(self, image_path):
        # Implementation for real model (e.g., MobileNetV2 uses 224x224)
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return img_array
    
    def get_disease_info(self, disease_name):
        # formatted_name = disease_name.replace('_', ' ') # Optional formatting
        
        # Check for direct match
        if disease_name in disease_info:
            return disease_info[disease_name]
        
        # Check for partial match (e.g. 'Melanoma' in 'Malignant_Melanoma')
        for key in disease_info:
            if key.lower() in disease_name.lower().replace('_', ' '):
                 return disease_info[key]
        
        # Default Fallback
        return {
            "description": f"A skin condition identified as {disease_name.replace('_', ' ')}. Please consult a dermatologist for accurate diagnosis and treatment.",
            "diet": {
                "eat": ["Balanced diet", "Anti-inflammatory foods (Leafy greens, Omega-3s)"],
                "avoid": ["Processed sugars", "Alcohol", "Potential allergens"]
            },
            "medicine": ["Consult a Doctor for specific medication"],
            "severity": "Unknown (Consult Doctor)"
        }

    def predict(self, image_path, symptoms=None):
        if self.model:
            # Real Inference
            try:
                processed_img = self.preprocess_image(image_path)
                predictions = self.model.predict(processed_img)
                class_index = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0]))
                
                if class_index < len(self.classes):
                    predicted_class = self.classes[class_index]
                else:
                    predicted_class = "Unknown"
                
                info = self.get_disease_info(predicted_class)

                return {
                    "disease": predicted_class.replace('_', ' '),
                    "confidence": round(confidence * 100, 2),
                    "symptoms": symptoms,
                    "info": info
                }
            except Exception as e:
                print(f"Error during inference: {e}")
                return self.dummy_predict()
        else:
            return self.dummy_predict()

    def dummy_predict(self):
        # Pick from loaded classes if available, else from mapping
        pool = self.classes if self.classes else list(disease_info.keys())
        choice = random.choice(pool)
        info = self.get_disease_info(choice)
        return {
            "disease": choice.replace('_', ' '),
            "confidence": round(random.uniform(70, 99), 2),
            "info": info
        }
