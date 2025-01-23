import numpy

"""
EPOCHS = 10 
BATCH_SIZE = 128 

steps = X.shape[0] // BATCH_SIZE

if steps * BATCH_SIZE < X.shape[0]: 
    steps += 1

for epoch in range(EPOCHS): 
    for step in range(steps): #During each step in each epoch, you are taking a slice of training data. 
        batch_X = X[step]

"""

'''
def spiral_data(samples, classes):

    input_data = numpy.zeros((samples, 2), dtype=numpy.float32)

    labels = numpy.zeros(samples, dtype=numpy.uint8)
    
    samples_per_class = samples // classes
    for class_num in range(classes):
        start_index = class_num * samples_per_class
        end_index = start_index + samples_per_class
        
        if class_num == classes - 1:
            end_index = samples

        r = numpy.linspace(0, 1, end_index - start_index) 
        theta = numpy.linspace(class_num * 4, (class_num + 1) * 4, end_index - start_index) + numpy.random.randn(end_index - start_index) * 0.1  # Angle
        
        input_data[start_index:end_index, 0] = r * numpy.sin(theta)  
        input_data[start_index:end_index, 1] = r * numpy.cos(theta)  
        labels[start_index:end_index] = class_num  

    return input_data, labels
'''

#NB: When referring to dimensions of matrices in the documented code below, I will use (r, c) where the first parameter is the number of rows, and the second the number of columns. The most common notation of this type is (number of samples, number of neurons) as, typically, a matrix in a neural network will have the same number of rows as the samples, and the same number of columns as neurons to store the state of each neuron for every sample. 

#Logical in-depth explanation of forward pass layer code (this forms the fundamentals of how data is passed through a neural network): 
#You essentially want to do the dot product of each input set and each weight set in all of their combinations.
#The data inputs you are given are the matrices 'inputs' and 'weights', which are 2D matrices where each list in the inputs matrix is a SAMPLE representing a feature set and each list in weights correspond to a NEURON'S weighted connections. 
#Therefore, you need to transpose the second matrix so that it fulfils the criteria to multiply two matrices as you must have the same number of rows in the first matrix as columns in the second matrix. 
#Therefore, before biases are added, you have an output matrix where each column represents the outputs of a single neuron in that layer. 
#Each bias in the bias matrix corresponds to a bias for an individual neuron in that layer. E.g. the first neuron in the layer has a bias of 2, the second 3 and so on. 
#Therefore, each bias needs to be added to every output in its corresponding neuron's column. 

class Layer: 
    """Represents a dense (fully-connected) layer in a neural network."""

    def __init__(self, number_inputs: int, number_neurons: int, weight_regulariser_l1: float = 0.0, weight_regulariser_l2: float = 0.0, bias_regulariser_l1: float = 0.0, bias_regulariser_l2: float = 0.0):
        """
        Initialises the layer with weights and biases
        
        Keyword arguments: 
        number_of_inputs (int): The number of inputs to the layer.
        number_of_neurons (int): The number of neurons in the layer. 
        weight_regulariser_l1 (float): NOTE FILL 
        weight_regulariser_l2 (float): NOTE FILL 
        bias_regulariser_l1 (float): NOTE FILL 
        bias_regulariser_l2 (float): NOTE FILL 
        """
        #NOTE: Can change weight initialisation factor to 0.1 or 0.01 if needed. See Glorot uniform distribution note from Keras devs. Important for weight initialisation. 
        self.weights = 0.01 * numpy.random.randn(number_inputs, number_neurons) #There's quite a lot of complicated maths behind why you multiply by 0.01. The general rule for initialising weights in a neural network is that smaller values work better as, if the initial weights are too large, then the gradients computed in the training process become too large (for gradient descent), leading to an unstable and slow convergence (to a local minima) as the starting values will be disproportionately large in comparison to the adjustments being made. Random.randn returns a matrix of the size specified of random values in the normal Gaussian distribution. 
        self.biases = numpy.zeros((1, number_neurons)) #Biasses are generally initialised to a set of 0s (based on the number of neurons in the layer). 
        self.weight_regulariser_l1 = weight_regulariser_l1
        self.weight_regulariser_l2 = weight_regulariser_l2
        self.bias_regulariser_l1 = bias_regulariser_l1
        self.bias_regulariser_l2 = bias_regulariser_l2

    def forward_pass(self, inputs: numpy.ndarray, training): 
        """
        Performs the forward pass through the layer. 
        
        Takes the input data, computes the dot product with the weights, adds the biases, and stores the results as the output. 
        
        Keyword arguments: 
        inputs (numpy.ndarray): Input matrix to the layer, shaped as (number of samples, number of neurons).
        """
        self.inputs = inputs #This is so that you are able to remember what the inputs are when calculating the partial derivative with respect to the weights during backpropagation. 
        self.output = numpy.dot(inputs, self.weights) + self.biases #What this dot product essentially does is transpose the weights matrix (second matrix) and then do normal matrix multiplication.  You do this order of parameters as you want the resulting array to be sample-related and not neuron-related. You want a list of layer outputs per sample rather than a list of neurons and their outputs sample-wise.

    def backward_pass(self, gradient_values: numpy.ndarray): 
        """
        Performs the backward pass through the layer. 

        Computes the gradients of the weights and biases based on the gradients from the next layer. It also calculates the gradient with respect to the inputs for backpropagation. 
        
        Keyword arguments: 
        gradient_values (numpy.ndarray): The gradients with respect to the outputs of this layer, shaped as (number of samples, number of neurons).
        """

        #Regularisation has changed the overall loss, which must be accounted for in the backpropagation of gradients. 

        self.gradient_weights = numpy.dot(self.inputs.T, gradient_values) #The gradient of the weights is the dot product of the transposed inputs and the gradient values. The weights array is formatted such that the rows contain weights related to each input (weights for all neurons for the given input). Therefore, you can multiply them by the gradient vector directly. 
        self.gradient_biases = numpy.sum(gradient_values, axis=0, keepdims=True) #The gradient of the biases can be found by summing the gradients across all the samples. Since the gradients are just a 2D list, 
        
        if self.weight_regulariser_l1 > 0: 
            gradient_l1 = numpy.ones_like(self.weights) #This creates a gradient array initialised to ones in the same dimensions as the weights array. 
            gradient_l1[self.weights < 0] = -1 #This sets all the values in the 1s array to -1 if their corresponding value in the weights array was negative. 
            self.gradient_weights += self.weight_regulariser_l1 * gradient_l1 #Calculation based on the derivative of L1 regularisation. 

        if self.weight_regulariser_l2 > 0: 
            self.gradient_weights += 2 * self.weight_regulariser_l2 * self.weights #Calculation based on the derivative of L2 regularisation. 

        if self.bias_regulariser_l1 > 0:
            gradient_l1 = numpy.ones_like(self.biases) #Same as described above for weight gradients, but for bias gradients. 
            gradient_l1[self.biases < 0] = -1
            self.gradient_biases += self.bias_regulariser_l1 * gradient_l1

        if self.bias_regulariser_l2 > 0: 
            self.gradient_biases += 2 * self.bias_regulariser_l2 * self.biases #Calculation based on the derivative of L2 regularisation. 

        self.gradient_inputs = numpy.dot(gradient_values, self.weights.T) #The gradient of the inputs is the dot product of the gradient valuse and the transposed weights. The weights must be transposed as you want to output to be the same shape as the gradient from the previous layer. 

class InputLayer: 

    def forward_pass(self, inputs, training): 
        self.output = inputs

#Explanation of dropout layer: 
#This type of layer disables some neurons, while the others pass through unchanged. 
#The idea here is to prevent a neural network from becoming too dependent on any neuron or for any neuron to be relied upon entirely in a specific instance (which can be common if a model overfits the training data). 
#It logistically works by randomly disabling neurons at a given rate during every forward pass, forcing the network to learn how to make accurate predictions with only a random part of neurons remaining. 
#To do this, a Bernoulli distribution is used (which is a special case of the binomial distribution where the number of trials is 1).


class DropoutLayer: 

    def __init__(self, rate): 
        self.rate = 1 - rate #We invert the inputted rate. For example, for a dropout rate of 0.1, you need a success rate of 0.9. 
    
    def forward_pass(self, inputs, training):
        
        self.inputs = inputs
        
        if not training: 
            self.output = inputs.copy()
            return

        self.binary_mask = numpy.random.binomial(1, self.rate, size=inputs.shape) / self.rate #np.random.binomial(n, p, size). n (number of trials), p (probability of success of the experiment), size (additional parameter size). 

        self.output = inputs * self.binary_mask

    def backward_pass(self, gradient_values): 
        self.gradient_inputs = gradient_values * self.binary_mask

class LinearActivation: 

    def forward_pass(self, inputs, training): 
        self.inputs = inputs
        self.output = inputs
    
    def backward_pass(self, gradient_values): 
        self.gradient_inputs = gradient_values.copy()

    def predictions(self, outputs): 
        return outputs


class ReluActivation: 
    """Applies the ReLU activation function"""

    def forward_pass(self, inputs: numpy.ndarray, training): 
        """
        Performs the forward pass through the ReLU activation function. 

        Applies the ReLU function, which sets all negative values in the input to zero, and preserves all other values. 

        Keyword arguments: 
        inputs (numpy.ndarray): Input matrix to the activation function, shaped as (number of samples, number of neurons) 
        """
        self.inputs = inputs #Stores the inputs as they need to be remembered for calculating the partial derivatives in backpropagation. 
        self.output = numpy.maximum(0, inputs) #This creates an output matrix where all the inputs bigger than or equal to zero keep their value, where all the inputs less than 0 become 0 (see reLU activation section in notebook for further info.). 

    def backward_pass(self, gradient_values: numpy.ndarray): 
        """
        Performs the backward pass through the ReLU activation function

        Calculates the gradient of the ReLU function, setting gradients to zero where the input was less than, or equal to, zero. 

        Keyword arguments: 
        gradient_values (numpy.ndarray): The gradients with respect to the oututs of this activation, shaped as (number of samples, number of neurons). 
        """
        self.gradient_inputs = gradient_values.copy() #This gradient values will need to be modified, so a copy is made to ensure that changes are only made to gradient inputs and not gradient values. 
        self.gradient_inputs[self.inputs <= 0] = 0 #When calculating the gradients of the ReLU function, all inputs less than or equal to zero should have a gradient of 0. 

    def predictions(self, outputs): 
        return outputs 
    
class SoftmaxActivation: 
    """Applies the softmax activation function"""
    
    def forward_pass(self, inputs: numpy.ndarray, training): 
        """
        Performs the forward pass through the softmax activation function. 

        Calculates and stores the probabilities of each possible output/outcome. 

        Keyword arguments: 
        inputs (numpy.ndarray): Input data to the activation function, shaped as (number of samples, number of neurons). 
        """
        self.inputs = inputs

        exp_values = numpy.exp(inputs - numpy.max(inputs,axis=1, keepdims=True)) #This essentially subtracts the max value of each row of the inputs. As explored in more detail in notebook, this prevents dead neurons and exploding values. 
        summed_exp_values = numpy.sum(exp_values, axis=1, keepdims=True) #A short explanation for each of the numpy parameters here: (1) axis=1 means that the sum will operate row-wise, which adds all values in a row together to get one row; (2) keepdiums ensures the dimensions of the matrix stays the same. The reason you want the sums for each row is that you want the outputs for each neuron to be condensed into a single output per neuron. 
        probabilities = exp_values/summed_exp_values #This divides each exponential neuron value by the corresponding sample exponential sum (each row being a sample). It is important to keep the same dimensions as before for probabilities so that you eventually get a weighted probability for each output class. 
        
        self.output = probabilities

    def backward_pass(self, gradient_values: numpy.ndarray): 
        """
        Performs the backward pass through the softmax activation function. 

        Calculates the gradient of the loss with respect to the inputs of the softmax function using a Jacobian matrix. 

        Keyword arguments: 
        gradient_values (numpy.ndarray): The gradients of the loss with respect to the outputs of the activation, shaped as (number of samples, number of classes).
        """
        self.gradient_inputs = numpy.empty_like(gradient_values) #This creates an empty array - which will become the gradient array - that has the same shape as the gradients received with respect to the outputs of the activation function (that need to have chain rule applied to get back to gradients with respect to inputs). 
        for index, (output, gradient_value) in enumerate(zip(self.output, gradient_values)): #This will iterate through self.output and predicted_values simulataneously. zip() pairs each output from the current layer with its corresponding gradient. Enumerate keeps track of the index of the current iteration and the paired values. 
            output = output.reshape(-1,1) #This reshapes each output into a column vector (hence the '1' parameter) with an unspecified number of rows.
            jacobian_matrix = numpy.diagflat(output) - numpy.dot(output, output.T) #Calculating the partial derivatives (for the Jacobian matrix) using the formula. 
            self.gradient_inputs[index] = numpy.dot(jacobian_matrix, gradient_value) #Calculating the final product of the Jacobian matrix and the gradient vector (from the passed in gradient array), storing the resulting vector as a row in the empty gradient_inputs array. 
            #After all iterations, the backward pass should have, for each sample, created a single partial derivative, which forms a 2D array of resulting vectors batch-wise. So, each row of the output matrix is the partial derivative of the softmax activation function for that sample with respect to the inputs. 

    def predictions(self, outputs): 
        return numpy.argmax(outputs, axis=1)

class SigmoidActivation: 

    def forward_pass(self, inputs, training): 
        self.inputs = inputs
        self.output = 1 / (1 + numpy.exp(-inputs))

    def backward_pass(self, gradient_values): 
        self.gradient_inputs = gradient_values * (1 - self.output) * self.output

    def predictions(self, outputs): 
        return (outputs > 0.5) * 1 #NOTE: WHY AM I MULTIPLYING BY 1 HERE???
    
class Loss: 
    """Base/parent class for the loss classes."""

    #Explanation of Regularisation: 
    #Regularisation methods are those which reduce generalisation error. 
    #The first forms of regularisation are L1 and L2 regularisation. They are used to calculate a number (called a penalty) added to the loss value to penalise the model for large weights and biases. 
    #This is because large weights might indicate that a neuron is attempting to memorise a data element. 

    #Explanation of L1 Regularisation: 
    #L1's regularisation penalty is the sum of all the absolute values for the weights and biases. 
    #This is a linear penalty as regularisation loss is directly proportional to parameter values.
    #L1 penalises small weights much more than L2 regularisation. This is why it is rarely used alone, and usually combined with L2.  
    #Lambda is a value that dictates the impact we want the regularisation penalty to carry. A higher value means a more significant penalty. 

    #Explanation of L2 Regularisation: 
    #L2's regularisation penalty is the sum of the squared weights and biases. 
    #This is a non-linear penalty because of the square function. 
    #L2 is commonly used as it does not affect small parameter values substantially and does not allow the model to grow weights too large.
    #Lambda is a value that dictates the impact we want the regularisation penalty to carry. A higher value means a more significant penalty. 

    def regularisation_loss(self): 
        """
        FILL IN
        
        """
        regularisation_loss = 0

        for layer in self.trainable_layers: 

            if layer.weight_regulariser_l1 > 0 : 
                regularisation_loss += layer.weight_regulariser_l1 * numpy.sum(numpy.abs(layer.weights)) #This performs the L1 regularisation calculation. Multiplying the sum of the absolute values of the weights by the L1's weight penalty value. 
            
            if layer.weight_regulariser_l2 > 0: 
                regularisation_loss += layer.weight_regulariser_l2 * numpy.sum(layer.weights * layer.weights) #This performs the L2 regularisation calculation. Multiplying the sum of the squares of the weights by the L2's weight penalty value. 
            
            if layer.bias_regulariser_l1 > 0 : 
                regularisation_loss += layer.bias_regulariser_l1 * numpy.sum(numpy.abs(layer.biases)) #This performs the L1 regularisation calculation. Multiplying the sum of the absolute values of the biases by the L1's bias penalty value. 
            
            if layer.bias_regulariser_l2 > 0: 
                regularisation_loss += layer.bias_regulariser_l2 * numpy.sum(layer.biases * layer.biases) #This performs the L2 regularisation calculation. Multiplying the sum of the squares of the biases by the L2's bias penalty value. 

        return regularisation_loss
    
    def remember_trainable_layers(self, trainable_layers): #This method essentiall 'tells' the loss object which layers in the Model object are trainable.
        self.trainable_layers = trainable_layers
    
    def calculate(self, output: numpy.ndarray, true_prediction_values: numpy.ndarray, regularisation=False) -> float: 
        """
        Calculates the loss value for the given predictions and correct prediction values. 
        
        Keyword arguments: 
        output (numpy.ndarray): The predicted output values, shaped as (number of samples, number of classes).
        true_prediction_values (numpy.ndarray): The true labels/prediction values, shaped as (number of samples, number of classes).
        NOTE ADD REGULARISATION!!!!, AMEND THIS FUNCTION DOCSTRING REALLY...

        Returns: 
        float: The computed average loss value across all samples. 
        """
        sample_losses = self.forward_pass(output, true_prediction_values) #The forward pass is called to make sample_losses, which is a 1D array where each value in the array is the log confidence value for a sample. 
        data_loss = numpy.mean(sample_losses) #This finds the mean loss value for all the samples. 
        
        self.accumulated_sum += numpy.sum(sample_losses) #Need to add this to calculate the sample-wise average. Mathematically, you can just sum the losses from all epoch batches and counts to calculate the mean value at the end of each epoch. 
        self.accumulated_count += len(sample_losses) 

        if not regularisation: 
            return data_loss
        
        return data_loss, self.regularisation_loss()
    
    def calculate_accumulated_mean(self, regularisation=False): 
        """
        FILL IN - Need a method that calculates mean at any point in the training of the neural network (e.g. could be between epochs). 
        
        """

        data_loss = self.accumulated_sum / self.accumulated_count 
        
        if not regularisation: 
            return data_loss
        
        return data_loss, self.regularisation_loss() #Regularisation loss does not need to be accumulated as it's calculated from the current state of layer parameters. 
    
    def reset_epoch(self): #This method is to reset variables for accumulated loss each time you start a new epoch (i.e. each time you pass through the training data again). 
        self.accumulated_sum = 0 
        self.accumulated_count = 0



class CategoricalCrossEntropyLoss(Loss): 
    """Categorical cross-entrop loss function. A child class of the Loss parent class"""

    def forward_pass(self, predicted_values: numpy.ndarray, true_prediction_values: numpy.ndarray) -> numpy.ndarray: 
        """
        Calculates the sample losses using categorical cross-entropy. 

        Adjusts the predicted values to prevent logarithm of zero and computes the negative log likelihood of the predicted values. 

        Keyword arguments: 
        predicted_values (numpy.ndarray): The predicted probabilities, shaped as (number of samples, number of classes).
        true_prediction_values (numpy.ndarray): The true prediction values/labels, either sparse or one-hot encoded. 
        
        Returns: 
        numpy.ndarray: The sample losses for each input, shape as (number of samples).
        """
        sample_number = len(predicted_values) #Gets the number of samples as the length of the predicted values array is the number of samples (rows). 

        predicted_values_adjusted = [] #Predicted values must be adjusted in the case that the neural network puts full confidence into the wrong class for a sample, then the loss calculation would involve calculating -log(0) which is not defined (asymptote, negative infinity!). Therefore, this adjustment will prevent loss from being exactly, making it a very small value instead, but won't make it a negative value (which it would if you tried to solve the problem by adding a very small value) and won't bias overall loss towards 1 as it is a very insignificant value. It essentially doesn't drag the mean towards any specific value, but prevents log(0). 
        for row in predicted_values: 
            adjusted_row = []
            for value in row: 
                adjusted_value = max(1e-7, min(value, 1-1e-7)) #Clipping values to avoid log(0). 
                adjusted_row.append(adjusted_value)
            predicted_values_adjusted.append(adjusted_row) 
        predicted_values_adjusted = numpy.array(predicted_values_adjusted)

        if len(true_prediction_values.shape) == 1: #This makes sure that the loss calculation works for both one-hot encoded labels and sparse labels. One-hot encoded labels are where all values, except for one, are encoded with 0s, and the correct label's position is a 1 for each sample. Sprase labels are where it only contains a list of the class' correct values (doesn't include the 0s from the other classes). Therefore, the one-hot encoding will be a multi-dimensional list as it must account for all rows and columns, where the sparse labelling will only be 1-dimensional (list of correct class values). 
            correct_confidences = predicted_values_adjusted[range(sample_number), true_prediction_values] #This takes advantage of a way numpy allows you to index arrays. You can use parameters which are two arrays of indices. The first allows us to filter the rows of data in the array (each relating to an individual sample's confidence distribution in this case) and the second array is used to determine the indexes of the elements you want (e.g. first element of the second array is the index of the xth row that you want, where x is the first element in the first array). There will be as many rows in the predicted values matrix as the number of samples, therefore, you can simplify by doing range(sample_number) which creates an array from 0 to sample_number-1. 
        elif len(true_prediction_values.shape) == 2: #If the true prediction values are two dimensional, then it is using a set of one-hot encoded vectors. 
            correct_confidences = numpy.sum(predicted_values_adjusted * true_prediction_values, axis=1) #This is a similar technique as before. Instead of filtering confidences based on the true prediction values, you multiply the confidence by the targets, zeroing out all the ones except the confidences for the correct result for each sample. You then perform a sum along the rows to preserve the number of rows (and, therefore, the number of samples) but get it to the point where there is one confidence value per row (and, therefore, one confidence value per sample) which is the confidence for that sample's correct 

        negative_log_confidences = -numpy.log(correct_confidences) #This follows the equations for confidence described further in notebook. 

        return negative_log_confidences #Returns the losses for each class, for each sample. 
    
    def backward_pass(self, gradient_values: numpy.ndarray, true_prediction_values: numpy.ndarray): 
        """
        Calculates the gradients for the backward pass of categorical cross-entropy loss. 

        Computes the gradients based on the predicted values and correct values for each class. It does this by applying to chain rule to propagate the gradient backwards. 

        Keyword arguments: 
        gradient_values (numpy.ndarray): The gradients, shaped as (number of samples, number of classes).
        true_prediction_values (numpy.ndarray): The true values/labels, formatted as either sparse or one-hot encoded.  
        """
        samples = len(gradient_values) #This finds the number of samples (as this is the number of rows of the predicted_values matrix). 
        labels = len(gradient_values[0]) #This finds the number of classes (finding the number of columns in the first row as the number of output values/classes should be the same for each sample). 
        
        if len(true_prediction_values.shape) == 1: #If the true prediction values are only one dimension (indicating that the labels are one-hot vectors/sparse vectors)
            true_prediction_values = numpy.eye(labels)[true_prediction_values] #This converts it into a one-dimensional matrix with the row that the correct label is on. e.g. if the correct label is 1 and there are three labels: [0,1,0]
        
        self.gradient_inputs = -(true_prediction_values)/gradient_values #This calculates the gradient based on the derivative of the categorical cross entropy loss function, using the relevant mathematical equation. 
        self.gradient_inputs = self.gradients/samples #This noramlises the gradient based on the number of samples 

class SoftmaxClassifier(): #Currently, we have calculated partial derivatives for both the Categorical Cross Entropy Loss and Softmax Activation functions. But it is actually possible to combine the derivatives of both equations and do it in a, computationally, much faster way. This way, we calculate the combined gradient of the loss and activation functions all in one. 
    """Combines the softmax activation and categorical cross-entropy loss classes"""
    '''
    NOTE: Don't technically need this section of the softmax classifier anymore I think....double check...
    def __init__(self): 
        """
        Initialises the softmax classifier with instances of the softmax activation and categorical cross-entropy loss classes.
        """
        self.activation = SoftmaxActivation() 
        self.loss = CategoricalCrossEntropyLoss() 

    def forward_pass(self, inputs: numpy.ndarray, true_prediction_values: numpy.ndarray) -> float: 
        """
        Performs the forward pass through the classifier. 

        Computes the softmax activation and then calculates the loss using the true prediction values (correct labels). 

        Keyword arguments: 
        inputs (numpy.ndarray): Input data to the classifier, shaped as (number of samples, number of classes). 
        true_prediction_values (numpy.ndarray): The correct classes/labels, formatted as either sparse or one-hot encoded.

        Returns: 
        float: The computed loss value based on the inputs for that sample. 
        """
        self.activation.forward_pass(inputs) 
        self.output = self.activation.output #This gets the output layer's probabilities from the activation function. 
        return self.loss.calculate(self.output, true_prediction_values) #This calculates and returns the loss value based on the output layer's predictions. 
    
    '''
        
    def backward_pass(self, gradient_values: numpy.ndarray, true_prediction_values: numpy.ndarray): 
        """
        Performs the backward pass through the classifier. 

        Calculates the gradients of the loss with respect to the inputs and then propagates back through the activation function in the same way. 
        This is done by combining the mathematical equations done in the backward passes of each respective function into a more computationally-efficient mathematical function. 

        Keyword arguments: 
        gradient_values (numpy.ndarray): The gradients of the loss with respect to the ouputs, shaped as (number of samples, number of classes). 
        true_prediction_values (numpy.ndarray): The correct labels/output classes, formatted as either sparse or one-hot encoded. 
        """
        sample_number = len(gradient_values) #This finds the number of samples (number of rows of the inputted matrix). 
        
        if len(true_prediction_values.shape) == 2: #This checks if the vectors are one-hot encoded vectors and, if they are, they are turned into discrete values. 
            true_prediction_values = numpy.argmax(true_prediction_values, axis=1)
            
        self.gradient_inputs = gradient_values.copy() #A copy of gradient_values must be made in order to ensure that you don't accidentally make alterations to the gradient value array.
        self.gradient_inputs[range(sample_number), true_prediction_values] -= 1 #This selects predicted probabilities corresponding to the true class labels for each sample. The minus 1 effectively calculates the gradient of the categorical cross entropy loss with respect to the softmax outputs.
        self.gradient_inputs = self.gradient_inputs / sample_number #This normalises the gradients by dividing by the number of samples. This ensures the gradients are averaged over the entire batch. 
        
class BinaryCrossEntropyLoss(Loss): 

    def forward_pass(self, predicted_values, true_prediction_values): 

        predicted_values_adjusted = numpy.clip(predicted_values, 1e-7, 1 - 1e-7)

        sample_losses = -(true_prediction_values * numpy.log(predicted_values_adjusted) + (1 - true_prediction_values) * numpy.log(1 - predicted_values_adjusted))
        samples_losses = numpy.mean(samples_losses, axis=-1)
        
        return sample_losses
    
    def backward_pass(self, gradient_values, true_prediction_values): 
        
        sample_number = len(gradient_values) 
        outputs = len(gradient_values[0])

        gradient_values_clipped = numpy.clip(gradient_values, 1e-7, 1 - 1e-7)

        self.gradient_inputs = -(true_prediction_values / gradient_values_clipped - (1 - true_prediction_values) / (1 - gradient_values_clipped)) / outputs

        self.gradient_inputs = self.gradient_inputs / sample_number

class MeanSquaredErrorLoss(Loss): 

    def forward_pass(self, predicted_values, true_prediction_values): 

        sample_losses = numpy.mean((predicted_values - true_prediction_values) ** 2, axis = -1) 
        return sample_losses
    
    def backward_pass(self, gradient_values, true_prediction_values): 

        sample_number = len(gradient_values) 
        outputs = len(gradient_values[0])

        self.gradient_inputs = -2 * (true_prediction_values - gradient_values) / outputs 
        self.gradient_inputs = self.gradient_inputs / sample_number

class MeanAbsoluteErrrorLoss(Loss): 

    def forward_pass(self, predicted_values, true_prediction_values): 

        sample_losses = numpy.mean(numpy.abs(true_prediction_values - predicted_values), axis = -1)

        return sample_losses
    
    def backward_pass(self, gradient_values, true_prediction_values): 
        
        sample_number = len(gradient_values) 
        outputs = len(gradient_values[0])

        self.gradient_inputs = numpy.sign(true_prediction_values - gradient_values) / outputs
        self.gradient_inputs = self.gradient_inputs / sample_number


#Logical in-depth explanation of optimisers (essential for understanding many of the variables in the training code below): 
#The process of adjsuting the weights and biasses using gradients to decrease loss is the job of the optimiser. The optimiser essentially works by subtracting the (learning rate * parameter_gradients) from the actual parameter values in order to adjust them (in small or large leaps based on the learning rate).
#Using 1 as a learning rate doesn't work very well as the steps being performed are far too large. You want to perform small steps - calculating the gradient, updating parameters by a small (negative, as you're subtracting) fraction of this gradient, and repeat this in a loop. 
#Using small steps ensures that you follow the direction of the steepest descent. 
#However, these steps musn't be too small, otherwise something called learning stagnation occurs (which is where the model gets stuck in a local minimum, rather than a global minimum).  
#Some optimisers also have a property called momentum. Momentum adds to the gradient what, in the physical world, we call inertia. Think of it as a ball rolling into a local minimum, and, with momentum, being more likely to roll out of that local minimum. It minimises the chances of bouncing around and getting stuck in a local minimum (as you're always trying to follow the gradient of steepest descent with gradients, even if it means getting stuck in a local minimum).
#By modifying the learning rate and the momentum, it's possible to significantly shorten the training time. 
#This careful modification can be done by starting at a learning rate too high, then gradually lowering the learning rate and increasing the momentum (which can be done through a learning rate decay - lowering the learning rate during training).
#The idea of the learning decay rate is to steadily 'decay' the learning rate per batch or per epoch. The decaying takes the step and the decaying ratio and multiplies them. Therefore, the further in training, the bigger the step is, the bigger result of the multiplication. We can take the reciprocal of this so that the further the training, the lower the value, and multiply the learning rate by it. 


class StochasticGradientDescentOptimiser: 
    """
    Stochastic Gradient Descent (SGD) optimiser function. One of the optimisers that could be implemented during training.

    The SGD optimiser adjusts weights and biases using gradients to minimise loss. It incorporates a decay rate that decreases the learning rate over time, allowing for the steps to decrease as the training progresses. 
    """

    def __init__(self, learning_rate: float =1.0, decay: float =0.0, momentum: float =0.0): #Setting a learning rate of 1 as the default for this optimiser, which will be decayed over time according to the decay rate. 
        """
        Initialises the SGD optimiser with initial/default values for the learning rate and decay rate. 

        Keyword arguments: 
        learning_rate (float): The initial learning rate, to be decayed over time by the decay rate. Set to 1 to begin with. 
        decay (float): The decay rate, which determines the rate at which the learning_rate will be decreased (i.e. the rate at which the 'steps' will be made smaller). Set to 0 to begin with. 
        momentum (float): The rate at which learning is accelerated. Set to 0 to begin with. 
        """
        self.learning_rate = learning_rate #This is the initial learning rate.
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0 #This is the number of 'steps' (as it's known in training), that the model has gone through. 
        self.momentum = momentum
    
    def pre_update_parameters(self): 
        """
        Updates the current learning rate based on the decay rate. 

        This method adjusts the learning rate according to teh decay factor and the number of iterations completed. As the number of iterations increases, and depending on the decay rate, learning rate decreases. 
        """   
        if self.decay: #This will only be performed if the decay rate is not 0. 
            self.current_learning_rate = self.learning_rate * (1.0/(1.0+self.decay*self.iterations)) #If there is a decay rate other than 0, this will update the self.current_learning_rate attribute using the relevant formula. The formula essentially multiplies the initial learning rate by 1/(the decay rate * the number of steps). The added 1 makes sure that the algorithm never raises the learning rate. For example, in the first step, we must divide 1 by the learning rate (0.001 for example), which would result in a current learning rate of 1000, which was not the intended result. The added 1 ensures that the result is a fraction of the starting learning rate. 
    
    def update_parameters(self, layer): 
        """
        Updates the parameters (weights and biases) of the given layer. 

        This method applies the calculated weight and bias updates using the current learning rate and, if applicable, momentum. 

        Keyword arguments: 
        layer (Layer): The layer whose parameters are to be updated.
        """
        if self.momentum: #This will only run if momentum is being implemented into the SGD optimiser. 
            
            if not hasattr(layer, 'weight_momentums'): 
                layer.weight_momentums = numpy.zeros_like(layer.weights) #If the layer does not contain momentum arrays, create them and fill them with zeroes. 
                layer.bias_momentums = numpy.zeros_like(layer.biases) #There needs to be a momentum array for both the weights and the biases of EACH layer. 

            weight_updates = (self.momentum * layer.weight_momentums) - (self.current_learning_rate * layer.gradient_weights) #Update in the way described in 'logical in-depth explanation of optimisers' 
            layer.weight_momentums = weight_updates

            bias_updates = (self.momentum * layer.bias_momentums) - (self.current_learning_rate * layer.gradient_biases) 
            layer.bias_momentums = bias_updates

        else: #This is the default update that will be happening if momentum isn't being used. Known as vanilla SGD parameter updates and normalisation. 
            weight_updates = -self.current_learning_rate * layer.gradient_weights
            bias_updates = -self.current_learning_rate * layer.gradient_biases

        layer.weights += weight_updates
        layer.biases += bias_updates

        #As described above, in each epoch (or step), the weights will be adjusted by subtracting the learning rate multiplied by the gradient weights. 
        #As described above, in each epoch (or step), the biases will be adjusted by subtracting the learning rate multiplied by the gradient biases. 

    def post_update_parameters(self): 
        """
        Increments the iteration count after parameters have been updated. This marks the next step. 

        This method is called after the updates to keep track of the number of 'steps' performed by the optimiser. 
        """
        self.iterations += 1

#Explanation of Adagrad optimiser: 
#It uses a per-parameter learning rate rather than a globally-shared learning rate. 
#The idea is to normalise updates made to FEATURES.
#Adagrad does this by keeping a history of previous updates - the bigger the sum of the UPDATES is (in either direction, positive or negative), the smaller updates are made further in training. 
#This allows less frequently updates parameters to keep up with changes.
#Code-wise, Adagrad is eventualised using a cache, which stores the squares of all the parameter gradients (squared as you want negative gradients to be made positive). 
#The parameter updates are then calculates as a function of the learning rate mulitplied by the gradient (which is the same process as SGD) and then divided by the square root of the cache plus the epsilon value. 
#The epsilon is a hyperparameter (pre-training control knob setting) that prevents division by 0. 
#As you are adding the squares of values together, and then square rooting (which gives a smaller value than just adding the two values together without squaring or square rooting), the cache value gradually grows more slowly. 
#Overall, the impact is that the learning rate for parameters with smaller gradients are decrased slowly, whilst the learning rate for parameters with larger gradients have their learning rates decreased faster. 


class AdagradOptimiser: 
    """
    Adagrad optimiser class. One of the optimisers that could be implemented during training.

    The Adagrad optimiser adapts the learning rate based on history of gradient information. This factors in larger steps for infrequent features and smaller steps for frequent features.
    """
    def __init__(self, learning_rate: float =1.0, decay: float =0.0): #Setting a learning rate of 1 as the default for this optimiser, which will be decayed over time according to the decay rate. 
        """
        Initialises the Adagrad optimise with initial/default values for learning rate and decay rate. 

        Keyword arguments: 
        learning_rate (float): The initial learning rate, to be decayed over time by the decay rate. Set to 1 to begin with. 
        decay (float): The decay rate, which determines the rate at which the learning_rate will be decreased (i.e. the rate at which the 'steps' will be made smaller). Set to 0 to begin with. 
        """
        self.learning_rate = learning_rate #This is the initial learning rate.
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0 #This is the number of 'steps' (as it's known in training), that the model has gone through. 
        self.EPSILON = 1e-7
    
    def pre_update_parameters(self): 
        """
        Updates the current learning rate based on the decay, if applicable. 

        This method adjusts the learning rate befor the parameter updates occur according to the relevant Adagrad equations and the decay parameter. 
        """  
        if self.decay: #This will only be performed if the decay rate is not 0. 
            self.current_learning_rate = self.learning_rate * (1.0/(1.0+self.decay*self.iterations)) #If there is a decay rate other than 0, this will update the self.current_learning_rate attribute using the relevant formula. The formula essentially multiplies the initial learning rate by 1/(the decay rate * the number of steps). The added 1 makes sure that the algorithm never raises the learning rate. For example, in the first step, we must divide 1 by the learning rate (0.001 for example), which would result in a current learning rate of 1000, which was not the intended result. The added 1 ensures that the result is a fraction of the starting learning rate. 
    
    def update_parameters(self, layer: numpy.ndarray): 
        """
        Updates the weights and biases of the given layer using the relevant Adagrad equations. 

        This method modifies the layer's weight and bias caches, applying Adagrad updates based on the current gradients of the layer. 

        Keyword arguments: 
        layer (Layer): The layer object whose parameters are to be updated. 
        """
        if not hasattr(layer, 'weight_cache'): #This will only perform on the first loop, when there are no weight caches or bias caches. 
            layer.weight_cache = numpy.zeros_like(layer.weights) 
            layer.bias_cache = numpy.zeros_like(layer.biases)

        #Performs correct cache update for the Adagrad optimiser. 
        layer.weight_cache += layer.gradient_weights ** 2
        layer.bias_cache += layer.gradient_biases ** 2

        #Vanilla SGD parameter updates and normalisation. 
        layer.weights += -self.current_learning_rate * layer.gradient_weights / (numpy.sqrt(layer.weight_cache) + self.EPSILON)
        layer.biases += -self.current_learning_rate * layer.gradient_biases / (numpy.sqrt(layer.bias_cache) + self.EPSILON)
        
    def post_update_parameters(self): 
        """
        Increments the iteration count after parameters have been updated. This marks the next step. 

        This method is called after the updates to keep track of the number of 'steps' performed by the optimiser. 
        """
        self.iterations += 1

#RMSProp (Root Mean Squared Propagation) is, like Adagrad, another adaptation of SGD (Stochasdtic Gradient Descent). It calculates an adaptive learning rate per parameter - like Adagrad - just in a different way.
#Explanation of the RMSProp optimiser: 
#Where Adagrad calculates the cache as cache += gradient ** 2, RMSProp calculates the cache as cache = rho * cache + (1 - rho) * gradient ** 2
#Adagrad adds a mecahnism similar to momentum (but NOT momentum) with a slightly different per-parameter adaptive learning rate (like Adagrad). This makes learning rate changes smoother than Adagrad.
#Instead of continually adding squared gradients to a cache like Adagrad, it uses a moving average of the cache. Each update to the cache retains a part of the 'old' cache, and als oupdates it with a fraction of the new, squared gradients. 
#The new hyperparameter is rho, which is the cache memory decay rate. Rho is there as because the optimiser carries so much momentum of gradient and adaptive learning rate updates, even small gradient updates are enough to keep it going, so a default learning rate of 1 is too big and will cause model instability. 

class RMSPropOptimiser: 
    """
    RMSProp optimiser class. One of the optimisers that could be implemented during training.

    The RMSProp optimiser calculates an adaptive learning rate per parameter - like Adagrad - just in a different way. 
    """
    def __init__(self, learning_rate: float =0.001, decay: float =0.0):
        """
        Initialises the RMSProp optimiser with a specified learning and decay rate. 

        Keyword arguments: 
        learning_rate (float): The initial learning rate. Set to 0.001 to begin with. 
        decay (float): The decay rate for the learning rate. Set to 0.0 to begin with. 
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.EPSILON = 1e-7
        self.RHO = 0.9 #The main additional parameter between Adagrad and RMSProp is rho. Rho is the cache memory decay rate. 

    def pre_update_parameters(self): 
        """
        Updates the current learning rate based on decay, if applicable. 

        This method adjusts the learning rate befor the parameter updates occur according to the relevant RMSProp equations and the decay parameter. 
        """
        if self.decay: 
            self.current_learning_rate = self.learning_rate * (1.0/ (1.0 + (self.decay * self.iterations))) #If decay is being implemented, like Adagrad, RMSProp calculates an adaptive learning rate PER parameter, just in a different way to Adagrad. Whilst AdaGrad calculates the cache as cache += gradient ** 2, RMSProp calculates the cache as cache = rho * cache + (1 - rho) * gradient ** 2. Instead of continually adding squared gradients to a cache (like in Adagrad), it uses a moving average of the cache. Each update to the cache retains a part of the cache and updates it with a fraction of the new, squared, gradients. 

    def update_parameters(self, layer: numpy.ndarray):
        """
        Updates the weights and biases of the given layer using the relevant RMSProp equations. 

        This method modifies the layer's weight and bias caches, applying RMSProp updates based on the current gradients of the layer. 

        Keyword arguments: 
        layer (Layer): The layer object whose parameters are to be updated. 
        """
        if not hasattr(layer, 'weight_cache'): #Like in the Adagrad optimiser, if the layer does not contain cache arrays (checking for weight cache is the same as checking for bias cache, having one means you have both, not having one means you have neither). If they do not exist, create them and fill them with zeros. 
            layer.weight_cache = numpy.zeros_like(layer.weights) 
            layer.bias_cache = numpy.zeros_like(layer.biases) 

        #Performs correct cache update for the RMSProp optimiser. 
        layer.weight_cache = (self.RHO * layer.weight_cache) + ((1-self.RHO) * layer.gradient_weights ** 2) 
        layer.bias_cache = (self.RHO * layer.bias_cache) + ((1-self.RHO) * layer.gradient_biases ** 2)

        #Vanilla SGD parameter updates and normalisation. 
        layer.weights += -self.current_learning_rate * layer.gradient_weights / (numpy.sqrt(layer.weight_cache) + self.EPSILON) 
        layer.biases += -self.current_learning_rate * layer.gradient_biases / (numpy.sqrt(layer.bias_cache) + self.EPSILON)

    def post_update_parameters(self): 
        """
        Increments the iteration count after parameters have been updated. This marks the next step. 

        This method is called after the updates to keep track of the number of 'steps' performed by the optimiser. 
        """
        self.iterations += 1

#Adam (Adaptive Momentum) is the most widely-used optimiser and built atop RMSProp, with the actual momentum concept from SGD added back in. This means momentum will be applied (as in SGD), and THEN a per-weight adaptive learning rate (as in RMSProp). 
#Explanation of the Adam optimiser: 
#The Adam optimiser also adds a bias correction mechanism. This is not the same as the layer's bias. 
#The bias correction mechanism compensates for the initial zeroed values before they warm up with initial steps. This correction is done by dividing momentum and caache values by 1-(beta^step). As step increases, beta^step approaches 0 and, thereofre 1-(beta^step) approaches 1 to get division by 1 as step increases (so that as training progresses parameter updates return to their typical values in later training). 
#Other than that, the code for the Adam optimiser is largely based on the RMSProp optimsier. The other additions are the cache seen from SGD along with beta 1 and beta 2 parameters. 
#NOTE TO ANNA: Didn't initially do this when coding but reviewing realise could probably adapt at least init features from RMSProp and then override methods with different mathematical equations. Could do this in edits. 

class AdamOptimiser: 
    """
    Adam optimiser class. One of the optimisers that could be implemented during training.

    The Adam optimiser applies an adaptive learning rate, as in RMSProp, and momentum, as in SGD. 
    """
    def __init__(self, learning_rate=0.001, decay=0.0):
        """
        Initialises the Adam optimiser with a specfiied learning and decay rate. 

        Keyword arguments: 
        learning_rate (float): The initial learnig rate. Set to 0.001 to begin with. 
        decay (float): The decay rate for the learning rate. set to 0.0 to begin with. 
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0 
        self.EPSILON = 1e-7
        self.BETA_1 = 0.9
        self.BETA_2 = 0.999

    def pre_update_parameters(self): 
        """
        Updates the current learning rate based on decay, if applicable. 

        This method adjusts the learning rate befor the parameter updates occur according to the relevant Adam equations and the decay parameter. 
        """
        if self.decay: 
            self.current_learning_rate = self.learning_rate * (1.0/(1.0 + self.decay * self.iterations))

    def update_parameters(self, layer: numpy.ndarray): 
        """
        Updates the weights and biases of the given layer using the relevant Adam equations. 

        This method modifies the layer's momentum and cache, applying the Adam updates based on the current gradients.

        Keyword arguments: 
        layer (Layer): The layer object whose parameters are to be updated. 
        """
        if not hasattr(layer, 'weight_cache'): #This will only perform on the first loop, when there are no weight caches or bias caches. 
            layer.weight_momentums = numpy.zeros_like(layer.weights) 
            layer.weight_cache = numpy.zeros_like(layer.weights) 
            layer.bias_momentums = numpy.zeros_like(layer.biases) 
            layer.bias_cache = numpy.zeros_like(layer.biases)

        #This updates the momentum with current gradients. 
        layer.weight_momentums = (self.BETA_1 * layer.weight_momentums) + ((1-self.BETA_1) * layer.gradient_weights)
        layer.bias_momentums = (self.BETA_1 * layer.bias_momentums) + ((1-self.BETA_1) * layer.gradient_biases)

        #This gets the correction momentum.
        weight_momentums_corrected = layer.weight_momentums / (1 - (self.BETA_1 ** (self.iterations + 1)))
        bias_momentums_corrected = layer.bias_momentums / (1 - (self.BETA_1 ** (self.iterations + 1)))

        #This updates the cache with squared current gradients (like in RMSProp). 
        layer.weight_cache = (self.BETA_2 * layer.weight_cache) + ((1-self.BETA_2) * layer.gradient_weights ** 2)
        layer.bias_cache = (self.BETA_2 * layer.bias_cache) + ((1-self.BETA_2) * layer.gradient_biases ** 2)

        #This gets the corrected cache. 
        weight_cache_corrected = layer.weight_cache / (1-(self.BETA_2 ** (self.iterations + 1)))
        bias_cache_corrected = layer.bias_cache / (1-(self.BETA_2 ** (self.iterations + 1)))

        #Vanilla SGD parameter updates and normalisation. 
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (numpy.sqrt(weight_cache_corrected) + self.EPSILON)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (numpy.sqrt(bias_cache_corrected) + self.EPSILON)

    def post_update_parameters(self): 
        """
        Increments the iteration count after parameters have been updated. This marks the next step. 

        This method is called after the updates to keep track of the number of 'steps' performed by the optimiser. 
        """
        self.iterations += 1

#Explanation of approaches to avoid overfiiting:
#When tested with some newly sampled testing data, the model performed less well roughly 80% accuracy as compared to previous 98%). This is sign of overfitting. 
#One option to prevent overfitting is to change the model's size. If the model is not learning at all, one solution might be to try a larger model. If the model is learning, but there's divergence between the training and testing data, the solution is a smaller model. Essentiall need to find the smallest model that still learns.
#Another way to avoid overfitting is to use regularisation techniques and a dropout layer. This employs something called hyperparameter searching, which tries different settings (e.g. layer sizes) to see if the model is learning something. If so, the model is trained longer, or fully. This picks the best set of hyperparameters and the smallest number of neurons (which makes it easier for the model to GENERALISE, NOT overfit). 

class Accuracy: 

    def calculate(self, predicted_values, true_prediction_values): 
        comparisons = self.compare(predicted_values, true_prediction_values) 
        
        accuracy = numpy.mean(comparisons)

        self.accumulated_sum += numpy.sum(comparisons) 
        self.accumulated_count += len(comparisons) 

        return accuracy 
    
    def calculate_accumulated(self): 

        accuracy = self.accumulated_sum / self.accumulated_count 

        return accuracy 
    
    def reset_epoch(self): 
        self.accumulated_sum = 0 
        self.accumulated_count = 0 
    
class RegressionAccuracy(Accuracy): 
    def __init__(self): 
        self.precision = None

    def initialise(self, true_prediction_values, reinit = False): #For regression, the init method will calculate an accuracy precision. Initialisation won't recalculate precision unless forced to do so by setting the reinit parameter to True. This allows for multiple use-cases, including setting self.precision independently, calling initialise whenever needed (e..g from outside of the model), and even calling it multiple times.
        if self.precision is None or reinit: 
            self.precision = numpy.std(true_prediction_values) / 250 
    
    def compare(self, predicted_values, true_prediction_values): 
        return numpy.absolute(predicted_values - true_prediction_values) < self.precision


class CategoricalAccuracy(Accuracy): 
    def __init__(self, binary=False): 
        self.binary = binary

    def initialise(self, true_prediction_values): 
        pass #No initialisation needs to be performed here, but empty method must exist as it's going to be called from train method of Model class and would otherwise cause error. 

    def compare(self, predicted_values, true_prediction_values): #This is the same as the accuracy calcuation for classification, just wrapped into a class with an additional switch parameter. This switch - binary - disables one-hot to spare label conversion, since this model always requires the true_prediction_values to be a 2D array that is NOT one-hot encoded. 
        
        if not self.binary and len(true_prediction_values.shape) == 2: 
            true_prediction_values = numpy.argmax(true_prediction_values, axis=1)
        return predicted_values == true_prediction_values


class Model: 

    def __init__(self): 
        self.layers = []
        self.softmax_classifier_output = None

    def add_layer(self, layer): 
        self.layers.append(layer)

    def set(self, loss, optimiser, accuracy):
        self.loss = loss
        self.optimiser = optimiser
        self.accuracy = accuracy

    def finalise_model(self): 
        
        self.input_layer = InputLayer()
        layer_count = len(self.layers) 
        self.trainable_layers = []

        for i in range(layer_count): 
            if i == 0: 
                self.layers[i].prev = self.input_layer #.prev and .next are dynamically assigned attributes here to help connections between layers show clearly, which is part of the reason a separate - although small - class was made for input layer. 
                self.layers[i].next = self.layers[i+1]
            
            elif i < layer_count - 1: 
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            
            else: 
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'): #Essentially checking if it is a trainable layer
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers) 

        if isinstance(self.layers[-1], SoftmaxActivation) and isinstance(self.loss, CategoricalCrossEntropyLoss): #This is checking to see if the softmax activation AND categorical cross entropy classes are being used. If so, the softmax classifier previously developed and described can be used to combine the two. 
            self.softmax_classifier_output = SoftmaxClassifier()

    def train_model(self, input_data, testing_data, epochs=1, print_step=1, validation_data=None): 
        
        self.accuracy.initialise(testing_data)

        for epoch in range(1, epochs+1): 
            
            output = self.forward_pass(input_data, training=True) 
            

            data_loss, regularisation_loss = self.loss.calculate(output, testing_data, regularisation=True)

            loss = data_loss + regularisation_loss

            predictions = self.output_layer_activation.predictions(output) 
            accuracy = self.accuracy.calculate(predictions, testing_data)

            self.backward_pass(output, testing_data)

            self.optimiser.pre_update_parameters()

            for layer in self.trainable_layers: 
                self.optimiser.update_parameters(layer)
            
            self.optimiser.post_update_parameters()

            if not epoch % print_step: 
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' + 
                      f'loss: {loss:.3f}, (' + 
                      f'data_loss: {data_loss:.3f},' +
                      f'reg_loss: {regularisation_loss:.3f}), '+
                      f'learning rate: {self.optimiser.current_learning_rate}')
                
            if validation_data is not None: 
                validation_input, validation_answer = validation_data

                output = self.forward_pass(validation_input, training=False) 

                loss = self.loss.calculate(output, validation_answer)

                predictions = self.output_layer_activation.predictions(output)

                accuracy = self.accuracy.calculate(predictions, validation_answer)

                if not epoch % print_step: 
                    print(f'validation, ' +
                        f'acc: {accuracy:.3f}, ' + 
                        f'loss: {loss:.3f}')
        
    def forward_pass(self, input_data, training): 
        self.input_layer.forward_pass(input_data, training)

        for layer in self.layers: 
            layer.forward_pass(layer.prev.output, training)
        
        return layer.output
    
    def backward_pass(self, output, testing_data): 

        if self.softmax_classifier_output is not None: #Need to check if the softmax classifier is being used in the model 

            self.softmax_classifier_output.backward_pass(output, testing_data) 

            self.layers[-1].gradient_inputs = self.softmax_classifier_output.gradient_inputs

            for layer in reversed(self.layers[:-1]): 
                layer.backward_pass(layer.next.gradient_inputs) 

        else: 

            self.loss.backward_pass(output, testing_data)
            
            for layer in reversed(self.layers): 
                layer.backward_pass(layer.next.gradient_inputs)


input_data, testing_data = spiral_data(samples=1000,classes=3)
validation_input, validation_answer =spiral_data(samples=100, classes=3)

model = Model()

model.add_layer(Layer(2,512, weight_regulariser_l2=5e-4, bias_regulariser_l2=5e-4))
model.add_layer(ReluActivation())
model.add_layer(DropoutLayer(0.1))
model.add_layer(Layer(512,3))
model.add_layer(SoftmaxActivation())

model.set(loss=CategoricalCrossEntropyLoss(), optimiser=AdamOptimiser(learning_rate=0.05, decay=5e-5), accuracy=CategoricalAccuracy())

model.finalise_model()

model.train_model(input_data, testing_data, validation_data=(validation_input, validation_answer), epochs=10000, print_step=100)
