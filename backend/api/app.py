import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from backend.translation.translator import Translator
from backend.neural_network.letter_recognition.asl_detector import predict_letter_from_base64

app = Flask(__name__)
CORS(app)

translator = Translator()


@app.route("/api/predictLetter", methods=["POST"])
def predict_letter(): 
    """
    Accept JSON payload containing base64-encoded image in the form: 
    {
        "image": "data:image/jpeg;base64,..."
    }
    
    Processes the image with ASL detector and returns predicted letter. 
    
    Returns: 
    char | str: If a character is returned, it is a predicted letter (e.g. "A"), if a string is returned it is one of "Stabilising...", "No Frame", "No Hand", "No Image Provided"
    """
    try: 
        data = request.get_json()
        if 'image' not in data: 
            return jsonify({'letter': 'No Image Provided'}), 400
        
        image_base64 = data['image']
        predicted_letter = predict_letter_from_base64(image_base64) 
        return jsonify({'letter': predicted_letter}), 200
    
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
    
    
    