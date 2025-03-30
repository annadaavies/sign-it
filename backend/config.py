import string
import os

### GENERAL APPLICATION CONSTANTS ###
#The number of frame predictions stored locally at a time. 
BUFFER_SIZE = 10 

#The number of consecutive predictions required for a consensus. 
CONSENSUS_THRESHOLD = 5 

#The theoretical maximum size of a dictionary from the custom Dictionary class.
MAX_DICTIONARY_SIZE = 1000


### CLOTHING MODEL CONSTANTS ###
#Base directory for the clothing recognition model and datasets.
CLOTHING_BASE_DIR = os.path.join("/Users", "anna", "desktop", "school", "A LEVEL - ANNA",
    "COMPUTER SCIENCE", "NEA", "backend", "neural_network", "clothing_recognition")

#Data directories (relative to CLOTHING_BASE_DIR).
CLOTHING_TRAIN_DIR = "train"
CLOTHING_TEST_DIR = "test"

#Image dimensions for clothing dataset images. 
CLOTHING_IMAGE_SIZE = (28, 28) 

#Training parameters.
CLOTHING_EPOCHS = 10 
CLOTHING_BATCH_SIZE = 128
CLOTHING_PRINT_STEP = 100 

#Path to save or load the trained clothing model. 
CLOTHING_MODEL_PATH = os.path.join(CLOTHING_BASE_DIR, "clothing.model")

#Clothing labels mapping.
CLOTHING_LABELS = {
    0: 'Top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Boot'
}

### ASL MODEL CONSTANTS ###
#Base directory for the ASL model and datasets. 
ASL_BASE_DIR = os.path.join("/Users", "anna", "desktop", "school", "A LEVEL - ANNA",
    "COMPUTER SCIENCE", "NEA", "backend", "neural_network", "letter_recognition")

#Data directories (absolute paths using ASL_BASE_DIR).
ASL_TRAIN_DATA_DIR = os.path.join(ASL_BASE_DIR, "train")
ASL_VAL_DATA_DIR = os.path.join(ASL_BASE_DIR, "test")

#Training parameters for ASL model. 
ASL_BATCH_SIZE = 25
ASL_EPOCH_STEPS = 10 #Number of steps per epoch. 
ASL_LEARNING_RATE = 0.0005

#Image dimensions for the ASL model. 
ASL_IMAGE_HEIGHT = 200 
ASL_IMAGE_WIDTH = 200

#File paths for saving/loading the ASL model. 
ASL_MODEL_PATH = os.path.join(ASL_BASE_DIR, "asl_model.keras")
LEARNING_CURVES_PATH = os.path.join(ASL_BASE_DIR, 'model_learning_curves.png')

#ASL labels mapping. 
ASL_LABELS = list(string.ascii_uppercase)
