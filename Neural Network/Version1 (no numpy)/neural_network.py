import numpy as np
import random
import math
from matrix import Matrix
import nnfs
from nnfs.datasets import spiral_data 

nnfs.init()

class Layer(): #all layers in this neural network are dense (aka. fully-connected) for the time being... 
    def __init__(self, number_of_inputs, number_of_neurons): 
        #bias and weight initialisation 
        self.weights = Matrix([[0.01 * random.gauss(0,1) for i in range(number_of_inputs)] for n in range(number_of_neurons)]) #lots of complicated maths behind why you multiply by 0.01. the general rule for initialising weights in a neural network is that smaller values work better as, if the INITIAL weights are too large, then the gradients computed in the training process become too large (for gradient descent), leading to an unstable and slow convergence (to a local minima) in the training process as the starting values will be disproportionately large in comparison to the adjustments being made. 
        self.biases = Matrix([[0 for i in range(number_of_neurons)]]) #biases generally initialised to a set of 0s (based on the number of neurons in the layer) 
        

    #method for defining how a forward pass works 
    def forward_pass(self, inputs): 
        #calculating the output values based on the inputs (either the original input or the input from a previous layer), weights for that layer and biases for that layer
        self.output = (Matrix(inputs).multiply(self.weights.transpose())).add(self.biases) #POTENTIAL ERROR, CHECK!!!!: works for now but not sure logic is secure. NOT SURE IF WEIGHTS MATRIX SHOULD BE TRANSPOSED HERE???  
        #You do this parameter order as you want the resulting array to be sample-related and not neuron-related. You want a list of layer outputs per each sample rather than a list of neurons and their outputs sample-wise. Here, you also must TRANSPOSE the weight matrix as the two matrices (inputs and weights) are currently row-related rather than row-to-column related.

'''
IN-DEPTH LAYER CODE EXPLANATION: 

You essentially want to do the dot product of each input set and each weight set in all of their combinations. The multiply function in the matrix class takes each row from the first array and each column from the second array and multiplies like that. 
However, the data input you are giving are the matrices 'inputs' and 'weights', which are 2d matrices where each list in the inputs matrix is a SAMPLE representing a feature set (see why neural netowkrs receive data in batches) and each list in the weights larger list correspond to A neuron's weighted connections
Therefore, the matrices are ROW-RELATED.
Therefore, you need to transpose the second matrix so that it fulfils the criteria to multiply two matrices as each ROW is related to the corresponding neuron COLUMN
Therefore, before biasses are added, you have an output matrix where each column represents the outputs of a single neuron in that layer.
Each bias in the bias list corresponds to a bias for a individual neuron in that layer. E.g. the first neuron in the layer has a bias of 2, the second 3 and so on. 
Therefore, each bias needs to be added to every output in its corresponding neuron's column. This is what requires the broadcasting. 
'''

#creating a class so that you can have separate activation objects for separate layers
class ReluActivation(): 

    def forward_pass(self, inputs): 
        self.output = [[] for list in inputs.matrix]
        for i in range(len(inputs.matrix)):
            for n in range(len(inputs.matrix[0])): 
                if inputs.matrix[i][n] > 0: 
                    self.output[i].append(inputs.matrix[i][n])
                else: 
                    self.output[i].append(0)

class SoftmaxActivation(): 
    
    def forward_pass(self, inputs): 
 
        for row in inputs.matrix: 
            max_value = max(row)
            for i in range(len(row)): 
                row[i] = row[i] - max_value #See point made in my notebook, this prevents dead neurons and exploding values!!

        E = 2.71828182846 
        exp_values = [[E**value for value in row] for row in inputs.matrix]
        sample_summed_exp_values = [] #You want the sums for each row as you want the outputs for each neuron to be condensed into a SINGLE OUTPUT PER NEURON
        for i in range(len(exp_values)): 
            sample_summed_exp_values.append([sum(exp_values[i])]) 

        probabilities = [[] for row in inputs.matrix] #It is important to keep the same dimensions so that you eventually get a weighted probability for EACH output class
        for i in range(len(exp_values)): 
            for value in exp_values[i]:
                probabilities[i].append(value/sample_summed_exp_values[i][0]) 
                
        self.output = Matrix(probabilities) 
        return self.output
    

class Loss: 

    def calculate(self, output, true_prediction_values): 

        sample_losses = self.forward_pass(output, true_prediction_values) #sample_losses will be a 1D array, where each value in the array is the log confidence value for A sample. 
        
        data_loss = sum(sample_losses)/len(sample_losses) #finding the mean loss value for ALL THE SAMPLES

        return data_loss
    

class CategoricalCrossEntropyLoss(Loss): 

    def forward_pass(self, predicted_values, true_prediction_values):
        sample_number = len(predicted_values.matrix)
        
        predicted_values_adjusted = [[] for row in predicted_values.matrix] #This is required as in the occasion that the neural network puts FULL confidence into the WRONG class for a sample, then the loss calculation will involve calculating -log(0) which is not defined (asymptote, negative infinity!). Therefore, this adjustment will prevent loss from being exactly 0, making it a very small value instead, but won't make it a negative value (which it would if you tried to solve the problem by adding a very small value) and won't bias overall loss towards 1 as it is a very insignificant value. It essentially doesn't drag the mean towards any specific value. 
        for i in range(len(predicted_values.matrix)): 
            for n in range(len(predicted_values.matrix[i])): 
                    adjusted_value = max(1e-7, (min(predicted_values.matrix[i][n], 1 - 1e-7)))
                    predicted_values_adjusted[i].append(adjusted_value)
        predicted_values_adjusted = Matrix(predicted_values_adjusted)


        if len(true_prediction_values.matrix) == 1: #This is making sure that the loss calculation works for BOTH one-hot encoded labels and sparse labels. One-hot encoded labels are where all values, except for one, are encoded with 0s, and the correct labels position is a 1 for each sample. Sparse labels are where it ONLY contains a list of the class' correct values (doesn't include the 0s from the other classes). Therefore, the one-hot encoding will be multi-dimensional list as it must account for all rows and columns, whereas the sparse labelling will only be 1-dimensional (list of correct class values) 
            correct_confidences = [predicted_values_adjusted.matrix[i][true_prediction_values.matrix[i]] for i in range(sample_number)] #This filters through in order to identify the correct classes for each sample, and extracting the value that the neural network predicted for that class. The output is a 1D array, where each element is the prediction the neural network gave for the CORRECT class for A sample, ideally if you had THREE samples, you want [1.0,1.0,1.0]

        elif len(true_prediction_values.matrix) > 1: 
            correct_confidences = []
            for i in range(len(true_prediction_values.matrix)): 
                target_index = true_prediction_values.matrix[i].index(1)
                correct_confidences.append(predicted_values_adjusted.matrix[i][target_index])  #This process is the same as MASKING values     

        negative_log_confidences = [-(math.log(value)) for value in correct_confidences]

        return(negative_log_confidences)
    


            

X,y = spiral_data(samples = 100, classes = 3) 

dense1 = Layer(2,3)

activation1 = ReluActivation()

dense2 = Layer(3,3) #The output layer must have the same number of inputs as the previous layer has OUTPUTS, and as many outputs as our data includes CLASSES

activation2 = SoftmaxActivation()

loss_function = CategoricalCrossEntropyLoss()

dense1.forward_pass(X)

activation1.forward_pass(dense1.output)

dense2.forward_pass(activation1.output) 

activation2.forward_pass(dense2.output)

print(activation2.output.matrix[:5])

loss = loss_function.calculate(activation2.output,y)






'''
#TWO LAYERS - hand-typed, random data
inputs = Matrix([[1, 2, 3, 2.5],
 [2., 5., -1., 2],
 [-1.5, 2.7, 3.3, -0.8]])
weights = Matrix([[0.2, 0.8, -0.5, 1],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]])
biases = Matrix([[2, 3, 0.5]])
weights2 = Matrix([[0.1, -0.14, 0.5],
 [-0.5, 0.12, -0.33],
 [-0.44, 0.73, -0.13]])
biases2 = Matrix([[-1, 2, -0.5]])
layer1_outputs = (inputs.multiply(weights.transpose())).add(biases)
layer2_outputs = (layer1_outputs.multiply(weights2.transpose())).add(biases2)
print(layer2_outputs)
'''

'''
#SINGLE LAYER - hand-typed, random data
my_inputs1 = [[1.0, 2.0, 3.0, 2.5],
 [2.0, 5.0, -1.0, 2.0],
 [-1.5, 2.7, 3.3, -0.8]]
my_weights1 = [[0.2, 0.8, -0.5, 1.0],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]
my_biases1 = [[2.0, 3.0, 0.5]]

matrix_weights = Matrix(my_weights1) 
matrix_inputs = Matrix(my_inputs1) 
matrix_biases = Matrix(biases2) 
my_layer1_outputs = (matrix_inputs.multiply(matrix_weights.transpose())).add(matrix_biases)
my_layer2_outputs = (matrix_inputs.multiply(matrix_weights.transpose())).add(matrix_biases) 
 #you do this parameter order as you want the resulting array to be sample-related and not neuron-related. You want a list of layer outputs per each sample rather than a list of neurons and their outputs sample-wise. Here, you also must TRANSPOSE the weight matrix as the two matrices (inputs and weights) are currently row-related rather than row-to-column related.

print(layer_outputs2)
'''