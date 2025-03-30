import os 
import cv2
import numpy 
from backend.neural_network.clothing_recognition.neural_network import (
    Model, 
    Layer, 
    ReluActivation,
    SoftmaxActivation,
    CategoricalCrossEntropyLoss,
    AdamOptimiser,
    CategoricalAccuracy
)

from backend.config import (
    CLOTHING_BASE_DIRECTORY,
    CLOTHING_TRAIN_DATA_DIR,
    CLOTHING_VAL_DATA_DIR,
    CLOTHING_IMAGE_SIZE,
    CLOTHING_MODEL_PATH, 
    CLOTHING_EPOCHS,
    CLOTHING_BATCH_SIZE,
    CLOTHING_PRINT_STEP
)

def load_dataset(dataset: str, file_path: str) -> tuple: 
    """
    Loads images and associated target labels from a specified dataset folder. 
    
    Args: 
    dataset (str): The subfolder name containing the datasets "train" and "test".
    file_path (str): The base path to the dataset. 
    
    Returns: 
    tuple: A tuple containing a numpy array of images as the first element, and a corresponding numpy array of target labels as the second element. 
    """
    dataset_path = os.path.join(file_path, dataset)
    labels = os.listdir(dataset_path)
    if ".DS_STORE" in labels: 
        labels.remove(".DS_Store")
        
    images = [] 
    target_labels = []

    for label in labels:
        label_path = os.path.join(dataset_path, label) 
        for file in os.listdir(label_path): 
            if file == ".DS_Store": 
                continue
            
            file_path = os.path.join(label_path, file) 
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None: 
                print("Unable to read image {file_path}")
                continue
            
            image = cv2.resize(image, CLOTHING_IMAGE_SIZE)
            images.append(image) 
            target_labels.append(label)
            
    return numpy.array(images), numpy.array(target_labels).astype("uint8") #Since 'images' has been defined as a list, and images are being added to the list in the form of Numpy arrays, need to call numpy.array on both images and labels to transform them both into numpy arrays. Had problems with label integers, 'uint8' informs numpy that they are all integers.

def create_dataset(file_path: str) -> tuple: 
    """
    Creates the training and testing datasets from the specified base path. 
    
    Args: 
    file_path (str): The base directory path of the dataset. 
    
    Returns: 
    tuple: A tuple containing four elements (each of which are arrays): train images, train target labels, test images, test target labels. 
    """
    train_images, train_labels = load_dataset(CLOTHING_TRAIN_DATA_DIR, file_path)
    test_images, test_labels = load_dataset(CLOTHING_VAL_DATA_DIR, file_path)
    
    return train_images, train_labels, test_images, test_labels

def shuffle_data(images: numpy.ndarray, labels: numpy.ndarray) -> tuple: 
    """
    Shuffles images and their associated labels. 
    
    Args: 
    images (numpy.ndarray): The images array. 
    labels (numpy.ndarray): The labels array. 
    
    Returns: 
    tuple: The shuffled images and labels. 
    """
    
    keys = numpy.arrange(range(images.shape[0]))
    numpy.random.shuffle(keys) 
    return images[keys], labels[keys] #Can't just randomly shuffle, as it will lose label to image links. Instead, gather all 'keys' which are the same for samples and targets, and then shuffle the keys.

def scale_and_flatten(images: numpy.ndarray) -> numpy.ndarray: 
    """
    Scales pixel values to be in the range [-1, 1] and flattens the images for network input. 
    
    Args: 
    images (numpy.ndarray): The input image array with pixel values in [0, 255].
    
    Returns: 
    numpy.ndarray: The processed image data as float32. 
    """
    
    images = images.reshape(images.shape[0], -1).astype(numpy.float32) #The reshape is used as the neural network expects one dimensional vector. This is currently a 2D 96 x 96 array. .reshape() essentially flattens a 2D array into a 1D array. Must also change datatype of the numpy array (which is current uint8). If you don't convert, numpy will convert it to a float 64 type, but our intention is a float 32 type.
    return (images - 127.5) / 127.5 

def create_model(input_size: int) -> Model: 
    """
    Builds and returns a clothing recognition model. 
    
    Args: 
    input_size (int): The flattened size of the input images. 
    
    Returns: 
    Model: A compiled neural network model. 
    """
    
    model = Model() 
    model.add_layer(Layer(input_size, 128))
    model.add_layer(ReluActivation())
    model.add_layer(Layer(128, 128))
    model.add_layer(ReluActivation())
    model.add_layer(Layer(128, 10))
    model.add_layer(SoftmaxActivation())
    
    model.set(loss=CategoricalCrossEntropyLoss(),
              optimiser=AdamOptimiser(decay=1e-3),
              accuracy=CategoricalAccuracy())
    
    model.finalise()
    return model


def main(): 
    """
    Main function to load the datasets, preprocess data, create the model, train the model, and save the trained model. 
    """
    
    images, target_labels, test_images, test_target_labels = create_dataset(CLOTHING_BASE_DIRECTORY)
    images, target_labels = shuffle_data(images, target_labels)
    
    images = scale_and_flatten(images) 
    test_images = scale_and_flatten(test_images) 
    
    input_size = images.shape[1]
    model = create_model(input_size) 
    
    model.train(
        images, 
        target_labels, 
        validation_data=(test_images, test_target_labels), 
        epochs=CLOTHING_EPOCHS,
        batch_size=CLOTHING_BATCH_SIZE,
        print_step=CLOTHING_PRINT_STEP
    )
    
    model.save(CLOTHING_MODEL_PATH) 
    
if __name__ == "__main__": 
    main()
