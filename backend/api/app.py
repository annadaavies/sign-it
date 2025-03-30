import tensorflow
from flask import Flask, request, jsonify
from flask_cors import CORS

from backend.translation.translator import Translator
from backend.neural_network.letter_recognition.asl_detector import ASLPredictor
from backend.neural_network.clothing_recognition.clothing_detector import ClothingPredictor


from backend.config import ASL_MODEL_PATH, ASL_LABELS, BUFFER_SIZE, CONSENSUS_THRESHOLD, ASL_IMAGE_HEIGHT, CLOTHING_MODEL_PATH, CLOTHING_LABELS

app = Flask(__name__)
CORS(app)

translator = Translator()
asl_model = tensorflow.keras.models.load_model(ASL_MODEL_PATH)
asl_predictor = ASLPredictor(model=asl_model, buffer_size=BUFFER_SIZE, consensus_threshold=CONSENSUS_THRESHOLD, processing_size=ASL_IMAGE_HEIGHT, asl_labels=ASL_LABELS)
clothing_predictor = ClothingPredictor(model_path=CLOTHING_MODEL_PATH, labels=CLOTHING_LABELS)

@app.route("/api/predictLetter", methods=["POST"])
def predict_letter(): 
    """
    Accept JSON payload containing base64-encoded image in the form: 
    {
        "image": "data:image/jpeg;base64,..."
    }
    
    Processes the image with ASL detector and returns the predicted letter. 
    
    Returns: 
    char | str: If a character is returned, it is a predicted letter (e.g. "A"), if a string is returned it is one of "Stabilising...", "No Frame", "No Hand", "No Image Provided"
    """
    try: 
        data = request.get_json()
        if 'image' not in data: 
            return jsonify({'letter': 'No image provided'}), 400
        
        image_base64 = data['image']
        predicted_letter = asl_predictor.predict_letter_from_base64(image_base64) 
        return jsonify({'letter': predicted_letter}), 200
    
    except Exception as exception: 
        app.logger.error(f"Prediction error: {exception}")
        return jsonify({'Error': 'Server error'}), 500
    
@app.route("/api/predictClothing", methods=["POST"])
def predict_clothing(): 
    """
    Accepts JSON payload containing base64-encoded image in the form: 
    {
        "image": "data:image/jpeg;base64,..."
    }
    
    Processes the image with clothing detector and returns the predicted clothing item. 
    """
    try: 
        data = request.get_json()
        if "image" not in data: 
            return jsonify({"item": "No image provided"}), 400
        
        image_base64 = data["image"]
        predicted_item = clothing_predictor.predict_item_from_base64(image_base64)
        
        return jsonify({"item": predicted_item}), 200
    
    except Exception as exception: 
        app.logger.error(f"Prediction error: {exception}")
        return jsonify({'Error': 'Server error'}), 500
    
    
@app.route('/api/translateText', methods=['POST'])
def translate_text(): 
    """
    Accepts a JSON payload containing text to translate into ASL in the form: 
    {
        "text": text, e.g. "Hello"
    }
    
    Returns: 
    dict: Dictionary of translated ASL signs. 
    """
    try: 
        data = request.get_json()
        if 'text' not in data or not data['text'].strip(): 
            return jsonify({'Error': 'No text provided'}), 400
        
        translation_dict = translator.translate_text(data['text']) 
    
        return jsonify({'signs': translation_dict.serialise()})
        
    except Exception as exception: 
        app.logger.error(f"Translation error: {exception}")
        return jsonify({'Error': 'Server error'}), 500
    
if __name__ == '__main__': 
    app.run(port=5000, debug=True)
    
    
    