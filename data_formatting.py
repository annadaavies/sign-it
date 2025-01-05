
import os
import cv2
import numpy
import matplotlib.pyplot as plt

IMAGE_PATH = "/Users/anna/desktop/school/A LEVEL - ANNA/COMPUTER SCIENCE/NEA/asl_alphabet_images"

def load_dataset(dataset, path): 
    labels = os.listdir(os.path.join(path, dataset, dataset))
    labels.remove('.DS_Store')

    X = []
    y = []

    for label in labels: 

        for file in os.listdir(os.path.join(path, dataset, dataset, label)):
        
            image = cv2.imread(os.path.join(path, dataset, dataset, label, file), cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (96,96))

            X.append(image) 
            y.append(label)

    return numpy.array(X), numpy.array(y)  #Since X and y have been defined as a list with images added as numpy arrays, it should be turned into a numpy array. 
    #NOTE: Fix numpy array thing for the labels. Labels are currently letters and should instead all be numbers. 
def create_dataset(path): 

    X,y = load_dataset('asl_alphabet_train', path)
    X_test, y_test = load_dataset('asl_alphabet_test', path)
    
    return X, y, X_test, y_test

X, y, X_test, y_test = create_dataset(IMAGE_PATH)

letter_to_value_conversion = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25,'del':26,'space':27}

#image_data = cv2.imread(path+'/A/A1.jpg', cv2.IMREAD_GRAYSCALE)
#image_data = cv2.resize(image_data, (96,96))

#plt.imshow(image_data, cmap='gray') 
#plt.show()

