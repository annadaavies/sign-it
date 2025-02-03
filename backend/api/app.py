from flask import Flask, request, jsonify
from flask_cors import CORS

from neural_network.load_and_predict import load_and_predict
from translation.translator import Translator

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    try: 
        image_file = request.files['image']
        image_bytes = image
        #preprocess function needs to be used here to convert to appropriate format. 
        #prediction function then should be called. 
        prediction = 'A' #arbitrary value for now to test interface. 
        return jsonify({'letter': prediction})
    
    except Exception: 
        return jsonify({'error': str(Exception)}), 500


@app.route('/api/translate', methods=['POST'])
def translate(): 
    try: 
        data = request.get_json()
        text = data.get('text', '')
        
        translator = Translator()
        translation_sequence = translator.translate_sentence(text) 
        
        return jsonify({'signs': translation_sequence})
        
    except Exception: 
        return jsonify({'error': str(Exception)}), 500
    
if __name__ == '__main__': 
    app.run(port=5000, debug=True)
    