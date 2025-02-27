from backend.neural_network.digit_recognition.neural_network import Model 
import numpy
import cv2
import os
from matplotlib import pyplot as plt

DIRECTORY_PATH = "/Users/anna/desktop/school/A LEVEL - ANNA/COMPUTER SCIENCE/NEA"

def process_image(image_path: str): 
    
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Error: Could not read image.")
        
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    threshold_value, thresholded_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    
    processed_image = cv2.resize(thresholded_image, (28, 28))
    
    processed_image_data = (processed_image.reshape(1, -1).astype(numpy.float32) - 127.5) / 127.5
    
    return processed_image_data
    

def load_and_predict(model_name: str, image_data) -> str: 
    
    ASL_LABELS = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V', 
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
    }
    
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', model_name)
    
    model = Model.load(model_path)
    
    confidences = model.predict(image_data)
    
    predictions = model.output_layer_activation.predictions(confidences)
    
    prediction = ASL_LABELS[predictions[0]]
    
    return prediction
    
"""
if __name__ == "__main__": 
    image_data = process_image("C_test.png")
    print(load_and_predict("asl.model", image_data))
"""

"""
DISPLAYING IMAGES TESTING: 

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Conversion')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(thresh, cmap='gray')
    plt.title('Processed Image (White Hand)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

"""


'''
#LOADING SAVED MODEL 
model = Model.load('asl_model')
model.evaluate(X_test, y_test)
'''

'''

# Label index to label name relation
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Read an image
image_data = cv2.imread('pants.png', cv2.IMREAD_GRAYSCALE)

# Resize to the same size as Fashion MNIST images
image_data = cv2.resize(image_data, (28, 28))

# Invert image colors
image_data = 255 - image_data

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Load the model
model = Model.load('fashion_mnist.model')

# Predict on the image
confidences = model.predict(image_data)

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
prediction = fashion_mnist_labels[predictions[0]]

print(prediction)
'''

'''
#PREDICTING WITH ASL MODEL
asl_labels = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V', 
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
}


image_data = cv2.imread(os.path.join("/Users/anna/desktop/school/A LEVEL - ANNA/COMPUTER SCIENCE/NEA_TEST/j_test.jpg"), cv2.IMREAD_GRAYSCALE)

image_data = cv2.resize(image_data, (64, 64))

threshold_value, image_data = cv2.threshold(image_data, 127, 255, cv2.THRESH_BINARY_INV)

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32)) / 255.0

# Load the model
model = Model.load('asl.model')

# Predict on the image
confidences = model.predict(image_data)

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
prediction = asl_labels[predictions[0]]

print(prediction)
'''