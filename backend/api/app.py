from flask import Flask, request, jsonify
from flask_cors import CORS

from backend.utilities.load_and_predict import *
from translation.translator import Translator

app = Flask(__name__)
CORS(app)
translator = Translator()

@app.route('/api/predict', methods=['POST'])
def predict():
    try: 
        if 'image' not in request.files: 
            return jsonify({'Error': 'No image provided.'}), 400
        
        image_file = request.files['image']
        temp_file_path = f"/temp/{image_file.filename}"
        image_file.save(temp_file_path)
        
        processed_image = process_image(temp_file_path)
        prediction = load_and_predict(processed_image)
         
        return jsonify({'letter': prediction})
    
    except Exception: 
        app.logger.error(f"Prediction error: {str(Exception)}")
        return jsonify({'Error': 'Server error'}), 500


@app.route('/api/translate', methods=['POST'])
def translate(): 
    try: 
        data = request.get_json()
        if 'text' not in data or not data['text'].strip(): 
            return jsonify({'Error': 'No text provided'}), 400
        
        translation_dict = translator.translate_text(data['text']) 
    
        return jsonify({'signs': translation_dict.serialise()})
        
    except Exception: 
        return jsonify({'error': str(Exception)}), 500
    
if __name__ == '__main__': 
    app.run(port=5000, debug=True)
    