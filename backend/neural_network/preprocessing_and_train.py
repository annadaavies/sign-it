from neural_network import *
import os
import cv2
import numpy

#Lots of unorganised preprocessing thoughts and attempts here. This file needs to eventually be where all data preprocessing is done, the train function of the model is called, and the trained model is saved. 
IMAGE_PATH = "/Users/anna/desktop/school/A LEVEL - ANNA/COMPUTER SCIENCE/NEA/backend/neural_network/asl_images"


def load_dataset(dataset, path): 
    labels = os.listdir(os.path.join(path, dataset))
    labels.remove(".DS_Store")
    print(labels) 

    images = [] 
    target_labels = []

    for label in labels:

        for file in os.listdir(os.path.join(path, dataset, label)):

            if file == ".DS_Store":
                continue

            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            image = cv2.resize(image, (28, 28))

            images.append(image)
            target_labels.append(label)

    return numpy.array(images), numpy.array(target_labels).astype("uint8")  # Since images has been defined as a list, and images are being added to the list in the form of Numpy arrays, need to call numpy.array on both images and labels to transform them both into numpy arrays. Had problems with label integers, 'uint8' informs numpy that they are all integers.


def create_dataset(path):

    images, target_labels = load_dataset("train", path)

    test_images, test_target_labels = load_dataset("test", path)

    return images, target_labels, test_images, test_target_labels


images, target_labels, test_images, test_target_labels = create_dataset(IMAGE_PATH)

# Data must be shuffled otherwise, in the ordered version of the training data, the model will keep getting very good at predicting a singular label, then the next, but never all at once. Therefore, shuffling data is used to prevent the model from becoming biased towards any single class.
keys = numpy.array(range(images.shape[0]))
print(images.shape[0])
print(keys)
numpy.random.shuffle(keys)
images = images[keys]  # We can't just randomly shuffle, as you'll lose label to image links. Instead, you can gather all 'keys' which are the same for samples and targets, and then shuffle the keys.
target_labels = target_labels[keys]


#The below scales the image data. Neural networks tend to work best with data in the range of either 0 to 1 or -1 to 1. Here, the image data is between 0 to 255. A common way is to subtract half the maximum of all pixel values (i.e. 255/2 = 127.5) then dividing by this same half to produce a range bounded by -1 and 1. 
images = (images.reshape(images.shape[0], -1).astype(numpy.float32) - 127.5) / 127.5 #The reshape is used as the neural network expects one dimensional vector. This is currently a 2D 96 x 96 array. .reshape() essentially flattens a 2D array into a 1D array. Must also change datatype of the numpy array (which is current uint8). If you don't convert, numpy will convert it to a float 64 type, but our intention is a float 32 type.
test_images = (test_images.reshape(test_images.shape[0], -1).astype(numpy.float32) - 127.5) / 127.5 

model = Model()

model.add_layer(Layer(images.shape[1], 128, weight_regulariser_l2=1e-4))
model.add_layer(ReluActivation())
model.add_layer(DropoutLayer(0.5))  

model.add_layer(Layer(128, 64, weight_regulariser_l2=1e-4))
model.add_layer(ReluActivation())
model.add_layer(DropoutLayer(0.3))  

model.add_layer(Layer(64, 26, weight_regulariser_l2=1e-4))
model.add_layer(SoftmaxActivation())

model.set(
    loss=CategoricalCrossEntropyLoss(),
    optimiser=AdamOptimiser(learning_rate=0.01, decay=1e-3),
    accuracy=CategoricalAccuracy(),
)

model.finalise()

model.train(
    images,
    target_labels,
    validation_data=(test_images, test_target_labels),
    epochs=10,
    batch_size=128,
    print_step=100,
)

#model.save("asl.model")

