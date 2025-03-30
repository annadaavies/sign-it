import os
import numpy
import tensorflow
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import (
    Input, 
    Dense,
    Conv2D,
    MaxPooling2D, 
    Flatten, 
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2

from backend.config import (
    ASL_TRAIN_DATA_DIR, 
    ASL_VAL_DATA_DIR, 
    ASL_BASE_DIR, 
    ASL_MODEL_PATH,
    LEARNING_CURVES_PATH,
    ASL_BATCH_SIZE, 
    ASL_IMAGE_HEIGHT, 
    ASL_IMAGE_WIDTH,
    ASL_EPOCH_STEPS, 
    ASL_LEARNING_RATE 
)


def random_zoom(image: tensorflow.Tensor, label: tensorflow.Tensor, zoom_range: tuple=(-0.1, 0.1)) -> tuple: 
    """
    Applies a random zoom to the image and then resizes it back to the original dimensions. 
    
    Args: 
    image (tensorflow.Tensor): The image tesnor to be zoomed. 
    label (tensorflow.Tensor): Corresponding label tensor. 
    zoom_range (tuple): Range for random zoom. Defaults to (-0.1, 0.1). 
    
    Returns: 
    tuple: The zoomed image (resized to its original shape) as the first element, and its corresponding label as the second element. 
    """
    zoom = numpy.random.uniform(zoom_range[0], zoom_range[1])
    
    new_height = int(ASL_IMAGE_HEIGHT * (1+zoom))
    new_width = int((ASL_IMAGE_HEIGHT) * (1+zoom))
    
    image = tensorflow.image.resize_with_crop_or_pad(image, new_height, new_width) 
    image = tensorflow.image.resize(image, [ASL_IMAGE_HEIGHT, ASL_IMAGE_WIDTH])
    
    return image, label

def augment(image: tensorflow.Tensor, label: tensorflow.Tensor) -> tuple: 
    """
    Applies a random left-right flip to the image. 
    
    Args: 
    image (tensorflow.Tensor): The image tensor to be augmented.
    label (tensorflow.Tensor): Corresponding label tensor. 
    
    Returns: 
    tuple: The augmented image as the first element, and its corresponding label as the second element. 
    """
    image = tensorflow.image.random_flip_left_right(image)
    return image, label
    
def create_datasets() -> tuple: 
    """
    Creates training and validation datasets from directory images.
    
    The images are rescaled to the [0,1] range and data augmentation is applied to the training set. 
    
    Returns: 
    tuple: The training and validation data 
    """
    
    train_dataset = tensorflow.keras.preprocessing.image_dataset_from_directory(ASL_TRAIN_DATA_DIR,
                                                                                 image_size=(ASL_IMAGE_HEIGHT, ASL_IMAGE_WIDTH),
                                                                                 batch_size = ASL_BATCH_SIZE, 
                                                                                 label_mode='categorical'
                                                                                ).map(lambda x,y: (x/255.0, y))
    
    validation_dataset = tensorflow.keras.preprocessing.image_dataset_from_directory(ASL_VAL_DATA_DIR,
                                                                                 image_size=(ASL_IMAGE_HEIGHT, ASL_IMAGE_WIDTH),
                                                                                 batch_size = ASL_BATCH_SIZE, 
                                                                                 label_mode='categorical'
                                                                                ).map(lambda x,y: (x/255.0, y))
    
    train_dataset = train_dataset.map(random_zoom, num_parallel_calls=tensorflow.data.AUTOTUNE)
    train_dataset = train_dataset.map(augment, num_parallel_calls=tensorflow.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tensorflow.data.AUTOTUNE)
    
    return train_dataset, validation_dataset
    

def create_model(input_shape: tuple=(ASL_IMAGE_HEIGHT, ASL_IMAGE_WIDTH, 3), num_classes: int=26) -> tensorflow.keras.Model: 
    """
    Creates and returns a CNN model.
    
    Returns: 
    tensorflow.keras.Model: A compiled Keras CNN model instance ready for training. 
    """
    
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=ASL_LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def load_or_create_model(model_path: str) -> tensorflow.keras.Model: 
    """
    Loads a pre-trained model from a given file if possible, otherwise creates a new model instance. 
    
    Args: 
    model_path (str): Path to the model file. 
    
    Returns: 
    tensorflow.keras.Model: The loaded or newly created model instance. 
    """
    
    try: 
        model = tensorflow.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
    except (ValueError, IOError) as e: 
        print(f"Could not load model from {model_path}: {e}")
        model = create_model()
    return model


def main(): 
    """
    Main function to load the datasets, create/load the model, train with callbacks, and save the trained model. 
    """
    os.makedirs(ASL_BASE_DIR, exist_ok=True)
    
    train_dataset, validation_dataset = create_datasets()
    model = load_or_create_model(ASL_MODEL_PATH)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='max', verbose=1)
    
    train_model = model.fit(
        train_dataset, 
        validation_data=validation_dataset, 
        epochs=ASL_EPOCH_STEPS, 
        callbacks=[early_stopping],
        verbose=1
    )
    
    model.save(ASL_MODEL_PATH)

if __name__ == "__main__": 
    main()