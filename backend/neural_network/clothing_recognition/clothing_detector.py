import cv2 
import numpy
import base64

from backend.neural_network.clothing_recognition.neural_network import Model
from backend.config import CLOTHING_MODEL_PATH, CLOTHING_LABELS

class ClothingPredictor: 
    """
    A predictor class for clothing items (Fashion MNIST dataset items). This class handles: 
    - Loading the clothing model from a '.model' file.
    - Preprocessing images (resizing, inverting, scaling).
    - Predicting the clothing category.
    """
    
    def __init__(self, model_path: str, labels: dict): 
        self.model = Model.load(model_path) 
        self.labels = labels
        
    def _preprocess_image(self, image_bgr: numpy.ndarray) -> "numpy.ndarray | None": 
        """
        Convert a BGR image to the properly scaled 28 x 28 grayscale array expected by the model. 
        This process involves: 
        - Converting to grayscale
        - Resizing to 28 x 28 
        - Inverting pixel values (given nature of training data with black backgrounds). 
        - Normalise pixel values to range -1 to 1. 
        
        Returns: 
        numpy.ndarray: 1D Array with shape (1, 784) suitable for model prediction, or None if the image cannot be processed.  
        """
        
        if image_bgr is None: 
            return None
        
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) 
        image_resized = cv2.resize(image_gray, (28,28))
        image_inverted = 255 - image_resized
        image_float = image_inverted.reshape(1, -1).astype(numpy.float32)
        image_scaled = (image_float - 127.5) / 127.5
        
        return image_scaled
    
    
    def predict_item_from_base64(self, image_b64: str) -> str: 
        """
        Accepts a base64-encoded image string (e.g. data:image/jpeg;base64,XXX....)
        Decodes and preprocesses image string, then predicts a clothing category using a trained model. 
        
        Args: 
        image_b64 (str): The base64-encoded image string.
        
        Returns: 
        str: The predicted clothing label. 
        """
        try: 
            base64_data = image_b64.split(",")[1]
        except IndexError: 
            return "No Frame"
    
        image_bytes = base64.b64decode(base64_data) 
        image_array = numpy.frombuffer(image_bytes, numpy.uint8) #This turns raw bytes into a Numpy array suitable for OpenCV. 
        
        frame_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame_bgr is None: 
            return "No Frame"
        
        preprocessed_image = self._preprocess_image(frame_bgr) 
        if preprocessed_image is None: 
            return "No Frame"
        
        confidences = self.model.predict(preprocessed_image) 
        predictions = self.model.output_layer_activation.predictions(confidences)
        predicted_label_index = predictions[0]
        
        print(f"Confidences: {confidences}")
        print(f"Predicted index: {predictions[0]}")
        
        return self.labels.get(predicted_label_index)
    
    
