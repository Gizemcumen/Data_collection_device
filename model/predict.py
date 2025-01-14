import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


class WallQualityPredictor:
    def __init__(self, model_path='model/saved_models/wall_quality_model.h5'):
        """Initialize predictor with trained model"""
        self.model = load_model(model_path)
        self.target_size = (224, 224)

    def preprocess_image(self, image_path):
        """Load and preprocess image for prediction"""
        # Load and resize image
        img = load_img(image_path, target_size=self.target_size)

        # Convert to array and preprocess
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        return img_array

    def predict(self, image_path):
        """Predict wall quality for given image"""
        # Preprocess image
        processed_img = self.preprocess_image(image_path)

        # Make prediction
        prediction = self.model.predict(processed_img)[0][0]

        # Convert to label (threshold at 0.5)
        label = "Good" if prediction >= 0.5 else "Bad"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        return {
            'label': label,
            'confidence': float(confidence),
            'raw_score': float(prediction)
        }


if __name__ == '__main__':
    # Example usage
    predictor = WallQualityPredictor()

    # Test on a single image
    test_image = 'model/data/test/sample.jpg'  # Replace with actual test image
    result = predictor.predict(test_image)
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")