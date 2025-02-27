import string
MODEL_PATH = "/Users/anna/desktop/school/A LEVEL - ANNA/COMPUTER SCIENCE/NEA/backend/neural_network/letter_recognition/asl_model.keras" #Path to TensorFlow ASL model. 
ASL_LABELS = list(string.ascii_uppercase) #Prediction labels for TensorFlow ASL model. 
BUFFER_SIZE = 10 #Number of frame predictions stored locally at a time. 
CONSENSUS_THRESHOLD = 5 #Threshold after which, if the model makes that consecutive number of letter predictions, the letter is officially predicted on display. 
PROCESSING_SIZE = 200 #Processing dimensions for hand image