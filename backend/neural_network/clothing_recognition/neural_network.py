import numpy
import pickle
import copy


class Layer:
    """A fully-connected dense neural network layer with regularisation support."""

    def __init__(
        self,
        number_inputs: int,
        number_neurons: int,
        weight_regulariser_l1: float = 0.0,
        weight_regulariser_l2: float = 0.0,
        bias_regulariser_l1: float = 0.0,
        bias_regulariser_l2: float = 0.0,
        ):
        """
        Initialise layer parameters with small, random (Gaussian) weights and zero biases. 

        Args:
        number_inputs (int): The number of input features to the layer.
        number_neurons (int): The number of neurons in the layer.
        weight_regulariser_l1 (float): L1 regularisation factor for the weight parameters.
        weight_regulariser_l2 (float): L2 regularisation factor for the weight parameters.
        bias_regulariser_l1 (float): L1 regularisation factor for the bias parameters.
        bias_regulariser_l2 (float): L2 regularisation factor for the bias parameters. 
        """
        self.weights = 0.01 * numpy.random.randn(number_inputs, number_neurons) #The weight initialisation factor can be 0.1 or 0.01. See Glorot uniform distribution note from Keras devs. 
        self.biases = numpy.zeros((1, number_neurons)) 
        
        #Set regularisation strengths
        self.weight_regulariser_l1 = weight_regulariser_l1
        self.weight_regulariser_l2 = weight_regulariser_l2
        self.bias_regulariser_l1 = bias_regulariser_l1
        self.bias_regulariser_l2 = bias_regulariser_l2

    def forward_pass(self, inputs: numpy.ndarray, training: bool):
        """
        Perform forward propagation through the layer. 

        Takes the input data, computes the dot product with the weights, adds the biases, and stores the results as the output.

        Args:
        inputs (numpy.ndarray): Input matrix to the layer (from the previous layer), shaped as (number of samples, number of neurons).
        training (bool): Boolean flag indicating training mode. 
        """
        
        self.inputs = inputs  #Remember inputs for later, when calculating the partial derivative with respect to the weights during backpropagation.
        self.output = (numpy.dot(inputs, self.weights) + self.biases)  #Transpose the weights matrix (second matrix) and perform matrix multiplication. Rresulting array is sample-related, not neuron-related. 

    def backward_pass(self, gradient_values: numpy.ndarray):
        """
        Perform backward propagation through the layer.

        Computes the gradients of the weights and biases based on the gradients from the next layer. It also calculates the gradient with respect to the inputs for backpropagation.

        Args:
        gradient_values (numpy.ndarray): Gradient of loss with respect to layer outputs, shaped as (number of samples, number of neurons).
        """

        self.gradient_weights = numpy.dot(self.inputs.T, gradient_values)  #Calculate parameter gradients using chain rule. 
        self.gradient_biases = numpy.sum(gradient_values, axis=0, keepdims=True)  #Find gradient of biases by summing the gradients across all the samples. 

        #Add regularisation gradients for weights
        if self.weight_regulariser_l1 > 0:
            gradient_l1 = numpy.ones_like(self.weights)  #Creates a gradient array initialised to ones in the same dimensions as the weights array.
            gradient_l1[self.weights < 0] = -1  #Set all values in the 1s array to -1 if their corresponding value in the weights array was negative.
            self.gradient_weights += (self.weight_regulariser_l1 * gradient_l1)  #Calculation based on the derivative of L1 regularisation.

        if self.weight_regulariser_l2 > 0:
            self.gradient_weights += (2 * self.weight_regulariser_l2 * self.weights)  #Calculation based on the derivative of L2 regularisation.

        #Add regularisation gradients for biases
        if self.bias_regulariser_l1 > 0:
            gradient_l1 = numpy.ones_like(self.biases)  #Same as described above for weight gradients, but for bias gradients.
            gradient_l1[self.biases < 0] = -1
            self.gradient_biases += self.bias_regulariser_l1 * gradient_l1

        if self.bias_regulariser_l2 > 0:
            self.gradient_biases += (2 * self.bias_regulariser_l2 * self.biases)  #Calculation based on the derivative of L2 regularisation.

        #Calculate final gradient for previous layer. 
        self.gradient_inputs = numpy.dot(gradient_values, self.weights.T)  #The gradient of the inputs is the dot product of the gradient values and the transposed weights. The weights must be transposed as you want to output to be the same shape as the gradient from the previous layer.

    def get_parameters(self) -> tuple:
        """
        Return current weights and biases as a tuple. 
        
        Returns: 
        tuple: A tuple where the first element is the weights matrix (shaped as (input dimensions, output dimensions)) and the biases vector (shaped as (1, output dimensions)). 
        """
        return self.weights, self.biases

    def set_parameters(self, weights: numpy.ndarray, biases: numpy.ndarray):
        '''
        Update layer parameters with new values. 
        
        Args: 
        weights: New weight matrix, shaped as (input_dimensins, output_dimensions).
        biases: New bias vector, shaped as (1, output_dimensions).
        '''
        self.weights = weights
        self.biases = biases


class InputLayer:
    """
    Input layer for the neural network. 
    
    This layers acts as a placeholder to pass the input values to subsequent layers without applying any transformations or computations. 
    """
    def forward_pass(self, inputs: numpy.ndarray, training: bool):
        """
        Perform forward propagation through the input layer. 
        
        Assigns received input data directly to output data. No modifications or computations are applied to the data in this layer. 
        
        Args:
        inputs (numpy.ndarray): The input data. 
        training (bool): training: Boolean flag indicating training mode. 
        """
        self.output = inputs


class DropoutLayer:
    """
    Regularisation layer that randomly zeros inputs during training. 
    
    Implements dropout regularisation by randomly masking a fraction of inputs during forward propagation and scaling output during training.
    """

    def __init__(self, dropout_rate: float):
        """
        Initialise dropout layer with specified dropout rate. 
        
        Args: 
        dropout_rate: The fraction of inputs to zero out (between 0 and 1).
        """
        self.dropout_rate = (1 - dropout_rate)  #We invert the inputted rate. For example, for a dropout rate of 0.1, you need a success rate of 0.9.

    def forward_pass(self, inputs: numpy.ndarray, training: bool):
        """
        Apply dropout mask during training, pass through otherwise. 
        
        During training, (1) Generate binary mask with probability dropout_rate (2) Mask input activations (3) Scale output by 1/dropout_rate
        
        Args: 
        inputs: Input activations from previous layer. 
        training: Boolean flag indicating training mode. 
        """
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        #Generate scaled dropout mask
        self.binary_mask = (numpy.random.binomial(1, self.dropout_rate, size=inputs.shape) / self.dropout_rate)  # np.random.binomial(n, p, size). n (number of trials), p (probability of success of the experiment), size (additional parameter size).

        self.output = inputs * self.binary_mask

    def backward_pass(self, gradient_values: numpy.ndarray):
        """
        Backpropagate gradients through dropout mask. 
        
        During the backward pass, gradients are masked using the same binary mask in forward pass. 
        During the forward pass, gradients are masked by 1/dropout_rate 
        
        Args: 
        gradient_values: Gradient of loss with respect to layer outputs. 
        """
        self.gradient_inputs = gradient_values * self.binary_mask


class LinearActivation:
    """Applies the Linear activation function"""

    def forward_pass(self, inputs: numpy.ndarray, training: bool):
        """
        Perform forward propagation through the linear activation function. 
        
        Passes through the inputs unchanged.
        
        Args: 
        inputs: Input matrix to the activation function, shaped as (number of samples, number of neurons).
        training: Boolean flag (unused in this activation). 
        """
        self.inputs = inputs
        self.output = inputs

    def backward_pass(self, gradient_values: numpy.ndarray):
        """
        Perform backward propagation through the linear activation function. 
        
        Passes through the gradients unchanged. 
        
        Args: 
        gradient_values: The gradients of loss with respect to layer outputs. 
        """
        self.gradient_inputs = gradient_values.copy()

    def predictions(self, outputs: numpy.ndarray) -> numpy.ndarray:
        """
        Returns outputs directly (no transformation needed for linear activation).
        
        Args: 
        outputs: Layer outputs after activation
        
        Returns: 
        numpy.ndarray: Unchanged outputs (linear activation doesn't modify predictions). 
        """
        return outputs


class ReluActivation:
    """
    Applies the ReLU (Rectified Linear Unit) activation function.
    
    In forward propagation, output = max(0, input)
    In backward propagation, derivative = 1 where input > 0, else 0.
    """

    def forward_pass(self, inputs: numpy.ndarray, training: bool):
        """
        Perform forward propagation through the ReLU activation function.

        Applies the ReLU function, which sets all negative values in the input to zero, and preserves all other values.

        Args:
        inputs: Input matrix to the activation function, shaped as (number of samples, number of neurons).
        training: Boolean flag (unused in this activation). 
        """
        self.inputs = inputs  #Store inputs as they need to be remembered for calculating the partial derivatives in backpropagation.
        self.output = numpy.maximum(0, inputs)  #Create an output matrix where all the inputs bigger than or equal to zero keep their value, where all the inputs less than 0 become 0.

    def backward_pass(self, gradient_values: numpy.ndarray):
        """
        Perform backward propagation through the ReLU activation function

        Calculates the gradient of the ReLU function, setting gradients to zero where the input was less than, or equal to, zero.

        Args:
        gradient_values (numpy.ndarray): The gradients with respect to the oututs of this activation, shaped as (number of samples, number of neurons).
        """
        self.gradient_inputs = (gradient_values.copy())  #Gradient values will later be modified, so a copy is made to ensure that changes are only made to gradient inputs and not gradient values.
        self.gradient_inputs[self.inputs <= 0] = 0 


    def predictions(self, outputs: numpy.ndarray) -> numpy.ndarray:
        """
        Return outputs directly (no transformation needed for ReLU). 
        
        Args: 
        outputs: Layer outputs after activation
        
        Returns: 
        numpy.ndarray: Unchanged outputs (ReLU doesn't modify predictions). 
        """
        return outputs


class SoftmaxActivation:
    """Applies the softmax activation function"""

    def forward_pass(self, inputs: numpy.ndarray, training: bool):
        """
        Perform forward propagation through the softmax activation function.

        Calculates and stores the probabilities of each possible output/outcome.

        Args:
        inputs: Input data to the activation function, shaped as (number of samples, number of neurons).
        training: Boolean flag (unused in this activation).
        """
        self.inputs = inputs

        exp_values = numpy.exp(inputs - numpy.max(inputs, axis=1, keepdims=True))  #Subtract the max value of each row of the inputs. Prevents dead neurons and exploding values.
        summed_exp_values = numpy.sum(exp_values, axis=1, keepdims=True) 
        probabilities = (exp_values / summed_exp_values)  #Divide each exponential neuron value by the corresponding sample exponential sum (each row being a sample). Keep probability dimensions to get a weighted probability for each output class.

        self.output = probabilities

    def backward_pass(self, gradient_values: numpy.ndarray):
        """
        Perform backward propagation through the softmax activation function.

        Calculate Jacobian-vector product to find the gradient of the loss with respect to the inputs of the softmax function.

        Args:
        gradient_values: The gradients of the loss with respect to the outputs of the activation, shaped as (number of samples, number of classes).
        """
        self.gradient_inputs = numpy.empty_like(gradient_values)  
        
        for index, (output, gradient_value) in enumerate(zip(self.output, gradient_values)):  #Iterate through self.output and predicted_values simulataneously. zip() pairs each output from the current layer with its corresponding gradient. Enumerate keeps track of the index of the current iteration and the paired values.
            
            output = output.reshape(-1, 1)  
            jacobian_matrix = numpy.diagflat(output) - numpy.dot(output, output.T)  #Calculate the partial derivatives (for the Jacobian matrix) using the formula.
            
            self.gradient_inputs[index] = numpy.dot(jacobian_matrix, gradient_value)  #Calculate the final product of the Jacobian matrix and the gradient vector (from the passed in gradient array). Store resulting vector as a row in the empty gradient_inputs array.
           
            #After all iterations, the backward pass should have, for each sample, created a single partial derivative, which forms a 2D array of resulting vectors batch-wise. So, each row of the output matrix is the partial derivative of the softmax activation function for that sample with respect to the inputs.

    def predictions(self, outputs: numpy.ndarray) -> numpy.ndarray:
        """
        Convert probabilities to class predictions
        
        Args: 
        outputs: Class probabilities from softmax activation
        
        Returns:
        numpy.ndarray: Array of class indices with highest probability. 
        """
        return numpy.argmax(outputs, axis=1)


class SigmoidActivation:
    """
    Applies the sigmoid activation function.
    
    In forward propagation, output = 1 / (1 + exp(-input)).
    In backward propagation, graidnet = output * (1 - output).
    """
    
    def forward_pass(self, inputs: numpy.ndarray, training: bool):
        """
        Perform forward propagation through the sigmoid activation function. 
        
        Args: 
        inputs: Input data to the activation function, shaped as (number of samples, number of neurons).
        training: Boolean flag (unused in this activation). 
        """
        self.inputs = inputs
        self.output = 1 / (1 + numpy.exp(-inputs))

    def backward_pass(self, gradient_values: numpy.ndarray):
        """
        Perform backpropagation through the sigmoid activation function. 
        
        Args: 
        gradient_values: The gradients of loss with respect to layer outputs. 
        """
        self.gradient_inputs = gradient_values * (1 - self.output) * self.output

    def predictions(self, outputs: numpy.ndarray) -> numpy.ndarray:
        """
        Convert probabilities to class predictions. 
        
        Args: 
        outputs: Class probabilities frm sigmoid activation. 
        
        Returns: 
        numpy.ndarray: Array of binary predictions based on 0.5 threshold. 
        """
        return (outputs > 0.5) * 1 


class Loss:
    """Base class for loss classes with regularisatin support."""
    
    def __init__(self): 
        """
        Initialises key loss variables. 
        """
        self.trainable_layers = []
        self.accumulated_sum = 0.0
        self.accumulated_count = 0
        
    def regularisation_loss(self) -> float:
        """
        Calculate L1 and L2 regularisation losses for all trainable layers. 
        
        Returns: 
        float: Total regularisation loss.
        """
        total_regularisation_loss = 0

        for layer in self.trainable_layers:

            #L1 regularisation for weights. 
            if layer.weight_regulariser_l1 > 0:
                total_regularisation_loss += layer.weight_regulariser_l1 * numpy.sum(numpy.abs(layer.weights)) 

            #L2 regularisation for weights. 
            if layer.weight_regulariser_l2 > 0:
                total_regularisation_loss += layer.weight_regulariser_l2 * numpy.sum(layer.weights * layer.weights)  

            #L1 regularisatin for biases.
            if layer.bias_regulariser_l1 > 0:
                total_regularisation_loss += layer.bias_regulariser_l1 * numpy.sum(numpy.abs(layer.biases)) 

            #L2 regularisation for biases.
            if layer.bias_regulariser_l2 > 0:
                total_regularisation_loss += layer.bias_regulariser_l2 * numpy.sum(layer.biases * layer.biases) 

        return total_regularisation_loss

    def remember_trainable_layers(self, trainable_layers: list):  
        """
        Store reference to the trainable layers for regularisation calculation.
        """
        self.trainable_layers = trainable_layers

    def calculate(
        self,
        output: numpy.ndarray,
        target_labels: numpy.ndarray,
        regularisation: bool = False
        ) -> "float | tuple":
        """
        Calculate loss value for the given predictions and targets.

        Args:
        output: The model's predicted output values, shaped as (number of samples, number of classes).
        target_labels: The ground truth values, shaped as (number of samples, number of classes).
        regularisation: Boolean flag that determines whether to include regularisation terms.

        Returns:
        float | tuple: Data loss (computed average loss value across all samples) or tuple of (data_loss, regularisation_loss). 
        """
        sample_losses = self.forward_pass(output, target_labels)  #Call forward pass to make sample_losses, which is a 1D array where each value in the array is the log confidence value for a sample.
        data_loss = numpy.mean(sample_losses) 

        self.accumulated_sum += numpy.sum(sample_losses)  #Calculate the SAMPLE-WISE average. Mathematically, you can just sum the losses from all epoch batches and counts to calculate the mean value at the end of each epoch.
        self.accumulated_count += len(sample_losses)

        if not regularisation:
            return data_loss

        return data_loss, self.regularisation_loss()

    def calculate_accumulated(self, regularisation: bool = False) -> "float | tuple":
        """
        Compute mean loss from accumulated values at any point in the training of the neural network (e.g. could be between epochs).
        
        Args: 
        regularisation:  Boolean flag that determines whether to include regularisation terms.
        
        Returns:
        float | tuple: Data loss (computed average loss value across all samples) or tuple of (data_loss, regularisation_loss). 
        """

        data_loss = self.accumulated_sum / self.accumulated_count

        if not regularisation:
            return data_loss

        return data_loss, self.regularisation_loss()  #Regularisation loss does not need to be accumulated as it's calculated from the current state of layer parameters.

    def reset_epoch(self):
        """
        Reset accumulated loss values for each new training epoch.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0


class CategoricalCrossEntropyLoss(Loss):
    """Categorical cross-entropy loss function (for multi-class classification problems). A child class of the Loss parent class."""

    def forward_pass(
        self, 
        predicted_probabilities: numpy.ndarray,
        true_labels: numpy.ndarray
        ) -> numpy.ndarray:
        """
        Calculate cross-entropy loss between predictions and true labels.

        Adjusts the predicted values to prevent logarithm of zero and computes the negative log likelihood of the predicted values.

        Args:
        predicted_probabilities: Softmax output predicted probabilities, shaped as (number of samples, number of classes).
        true_labels: Ground truth labels, either sparse or one-hot encoded.

        Returns:
        numpy.ndarray: Array of loss values per sample, shaped as (number of samples).
        """
        sample_number = len(predicted_probabilities) 

        #Probabilities adjusted in the case that the neural network puts full confidence into the wrong class for a sample, then the loss calculation would involve calculating -log(0) which is not defined (asymptote, negative infinity!). Therefore, this adjustment will prevent loss from being exactly, making it a very small value instead, but won't make it a negative value (which it would if you tried to solve the problem by adding a very small value) and won't bias overall loss towards 1 as it is a very insignificant value. It essentially doesn't drag the mean towards any specific value, but prevents log(0).
        
        clipped_probabilities = numpy.clip(predicted_probabilities, 1e-7, 1 - 1e-7)

        if len(true_labels.shape) == 1:  #Handle both one-hot encoded labels and sparse labels. One-hot encoded labels are where all values, except for one, are encoded with 0s, and the correct label's position is a 1 for each sample. Sprase labels are where it only contains a list of the class' correct values (doesn't include the 0s from the other classes). Therefore, the one-hot encoding will be a multi-dimensional list as it must account for all rows and columns, where the sparse labelling will only be 1-dimensional (list of correct class values).
            correct_confidences = clipped_probabilities[range(sample_number), true_labels]  #You can use parameters which are two arrays of indices. The first allows us to filter the rows of data in the array (each relating to an individual sample's confidence distribution in this case) and the second array is used to determine the indexes of the elements you want (e.g. first element of the second array is the index of the xth row that you want, where x is the first element in the first array). There will be as many rows in the predicted values matrix as the number of samples, therefore, you can simplify by doing range(sample_number) which creates an array from 0 to sample_number-1.
        
        elif len(true_labels.shape) == 2:  #If true labels are two dimensional, then it is using a set of one-hot encoded vectors.
            correct_confidences = numpy.sum(clipped_probabilities * true_labels, axis=1)  #Similar technique as before. Instead of filtering confidences based on the true prediction values, you multiply the confidence by the targets, zeroing out all the ones except the confidences for the correct result for each sample. You then perform a sum along the rows to preserve the number of rows (and, therefore, the number of samples) but get it to the point where there is one confidence value per row (and, therefore, one confidence value per sample) which is the confidence for that sample's correct

        negative_log_confidences = -numpy.log(correct_confidences)  

        return negative_log_confidences  #Return the losses for each class, for each sample.

    def backward_pass(
        self, 
        gradient_values: numpy.ndarray, 
        true_labels: numpy.ndarray
        ):
        """
        Calculate gradient of categorical cross-entropy loss with respect to inputs.

        Computes the gradients based on the predicted values and true values for each class. It does this by applying to chain rule to propagate the gradient backwards.

        Args:
        gradient_values: The gradients (from subsequent layers), shaped as (number of samples, number of classes).
        true_labels: Ground truth labels, either sparse or one-hot encoded.
        """
        samples = len(gradient_values)  #Find the number of samples (as this is the number of rows of the predicted_values matrix).
        labels = len(gradient_values[0])  #Find the number of classes (finding the number of columns in the first row as the number of output values/classes should be the same for each sample).

        if (len(true_labels.shape) == 1):  # If the true prediction values are only one dimension (indicating that the labels are one-hot vectors/sparse vectors)
            true_labels = numpy.eye(labels)[true_labels]  # This converts it into a one-dimensional matrix with the row that the correct label is on. e.g. if the correct label is 1 and there are three labels: [0,1,0]

        self.gradient_inputs = -true_labels / gradient_values  # This calculates the gradient based on the derivative of the categorical cross entropy loss function, using the relevant mathematical equation.
        self.gradient_inputs = self.gradients / samples  # This noramlises the gradient based on the number of samples


class BinaryCrossEntropyLoss(Loss):
    """Binary cross entropy loss function (for binary classification problems). A child class of the Loss parent class."""

    def forward_pass(self, predicted_probabilities: numpy.ndarray, true_labels: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate binary cross-entropy loss between predictions and true labels. 
        
        Args: 
        predicted_probabilities: Model output predicted probabilities, shaped as (number of samples, number of classes)
        true_labels: Ground truth labels, either sparse or one-hot encoded. 
        
        Returns: 
        numpy.ndarray: Array of loss values per sample, shaped as (number of samples).
        """

        predicted_probabilities = numpy.clip((true_labels), 1e-7, 1 - 1e-7)

        sample_losses = -(true_labels * numpy.log(predicted_probabilities) + (1 - true_labels) * numpy.log(1 - predicted_probabilities))
        samples_losses = numpy.mean(samples_losses, axis=-1)

        return sample_losses

    def backward_pass(self, gradient_values: numpy.ndarray, true_labels: numpy.ndarray):
        """
        Calculate gradient of binary cross entropy loss with respect to inputs. 
        
        Computes the gradients based on the predicted values and true values for each class. It does this by applying to chain rule to propagate the gradient backwards.
        
        Args: 
        gradient_values: The gradients (from subsequent layers), shaped as (number of samples, number of classes).
        true_labels: Ground truth labels, either sparse or one-hot encoded.
        """
        sample_number = len(gradient_values)
        outputs = len(gradient_values[0])

        clipped_gradients = numpy.clip(gradient_values, 1e-7, 1 - 1e-7)

        self.gradient_inputs = -(true_labels / clipped_gradients - (1 - true_labels) / (1 - clipped_gradients)) / outputs
    
        self.gradient_inputs = self.gradient_inputs / sample_number


class MeanSquaredErrorLoss(Loss):
    """Mean squared error loss function for regression tasks. A child class of the Loss parent class."""

    def forward_pass(self, predicted_probabilities: numpy.ndarray, true_labels: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate mean squared error loss. 
        
        Args: 
        predicted_probabilities: Model output predicted probabilities, shaped as (number of samples, number of classes)
        true_labels: Ground truth labels, either sparse or one-hot encoded. 
        
        Returns: 
        numpy.ndarray: Array of loss values per sample, shaped as (number of samples).
        """
        
        sample_losses = numpy.mean((predicted_probabilities - true_labels) ** 2, axis=-1)
        return sample_losses

    def backward_pass(self, gradient_values: numpy.ndarray, true_labels: numpy.ndarray):
        """
        Calculate gradient of mean squared loss with respect to inputs. 
        
        Computes the gradients based on the predicted values and true values for each class. It does this by applying to chain rule to propagate the gradient backwards.
        
        Args: 
        gradient_values: The gradients (from subsequent layers), shaped as (number of samples, number of classes).
        true_labels: Ground truth labels, either sparse or one-hot encoded.
        """
        sample_number = len(gradient_values)
        outputs = len(gradient_values[0])

        self.gradient_inputs = -2 * (true_labels - gradient_values) / outputs
        self.gradient_inputs = self.gradient_inputs / sample_number


class MeanAbsoluteErrorLoss(Loss):
    """Mean absolute error loss for regression tasks. A child class of the Loss parent class."""

    def forward_pass(self, predicted_probabilities: numpy.ndarray, true_labels: numpy.ndarray):

        sample_losses = numpy.mean(numpy.abs(true_labels - predicted_probabilities), axis=-1)
        return sample_losses

    def backward_pass(self, gradient_values: numpy.ndarray, true_labels: numpy.ndarray):

        sample_number = len(gradient_values)
        outputs = len(gradient_values[0])

        self.gradient_inputs = numpy.sign(true_labels - gradient_values) / outputs
        self.gradient_inputs = self.gradient_inputs / sample_number


class SoftmaxClassifier:  #Calculated partial derivatives for both the Categorical Cross Entropy Loss and Softmax Activation functions. It is possible to combine the derivatives of both equations and do it in a, computationally, much faster way. 
    """Combined softmax activation and categorical cross-entropy loss classes for optimised gradient calculation."""

    def backward_pass(self, gradient_values: numpy.ndarray, true_labels: numpy.ndarray):
        """
        Calculate optimised gradient for combined softmax activation and cross-entropy loss. 

        Computes the gradients of the loss with respect to the inputs and then propagates back through the activation function in the same way.
        This is done by combining the mathematical equations done in the backward passes of each respective function into a more computationally-efficient mathematical function.

        Args:
        gradient_values: The gradients (from subsequent layers), shaped as (number of samples, number of classes).
        true_labels: Ground truth labels, either sparse or one-hot encoded.
        """
        sample_number = len(gradient_values)  # This finds the number of samples (number of rows of the inputted matrix).

        if len(true_labels.shape) == 2:  # This checks if the vectors are one-hot encoded vectors and, if they are, they are turned into discrete values.
            true_labels = numpy.argmax(true_labels, axis=1)

        self.gradient_inputs = (gradient_values.copy())  # A copy of gradient_values must be made in order to ensure that you don't accidentally make alterations to the gradient value array.
        self.gradient_inputs[range(sample_number), true_labels] -= 1  # This selects predicted probabilities corresponding to the true class labels for each sample. The minus 1 effectively calculates the gradient of the categorical cross entropy loss with respect to the softmax outputs.
        self.gradient_inputs = (self.gradient_inputs / sample_number)  # This normalises the gradients by dividing by the number of samples. This ensures the gradients are averaged over the entire batch.


class StochasticGradientDescentOptimiser:
    """
    Stochastic Gradient Descent (SGD) optimiser function. One of the optimisers that could be implemented during training.

    Adjusts weights and biases using gradients to minimise loss. It incorporates a decay rate that decreases the learning rate over time, allowing for the steps to decrease as the training progresses.
    """

    def __init__(
        self, 
        learning_rate: float = 1.0, 
        decay: float = 0.0, 
        momentum: float = 0.0
        ): 
        """
        Initialise the SGD optimiser parameters.

        Keyword arguments:
        learning_rate (float): The initial learning rate, to be decayed over time by the decay rate. Set to 1 to begin with.
        decay (float): The decay rate, which determines the rate at which the learning_rate will be decreased (i.e. the rate at which the 'steps' will be made smaller). Set to 0 to begin with.
        momentum (float): The rate at which learning is accelerated. Set to 0 to begin with.
        """
        self.learning_rate = learning_rate  # This is the initial learning rate.
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0  # This is the number of 'steps' (as it's known in training), that the model has gone through.
        self.momentum = momentum

    def pre_update_parameters(self):
        """
        Updates the current learning rate based on the decay rate.

        This method adjusts the learning rate according to teh decay factor and the number of iterations completed. As the number of iterations increases, and depending on the decay rate, learning rate decreases.
        """
        if self.decay: 
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))  #Formula multiplies initial learning rate by 1/(the decay rate * the number of steps). The added 1 makes sure that the algorithm never raises the learning rate. For example, in the first step, we must divide 1 by the learning rate (0.001 for example), which would result in a current learning rate of 1000, which was not the intended result. The added 1 ensures that the result is a fraction of the starting learning rate.

    def update_parameters(self, layer: Layer):
        """
        Update layer parameters (weights and biases) using SGD with momentum.

        Applies calculated weight and bias updates using the current learning rate and, if applicable, momentum.

        Args:
        layer: The layer whose parameters are to be updated.
        """
        if self.momentum:  
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = numpy.zeros_like(layer.weights)  
                layer.bias_momentums = numpy.zeros_like(layer.biases)  #Create a momentum array for both the weights and the biases of EACH layer.

            #Update weights with momentum. 
            weight_updates = (self.momentum * layer.weight_momentums) - (self.current_learning_rate * layer.gradient_weights)
            layer.weight_momentums = weight_updates

            #Update biases with momentum. 
            bias_updates = (self.momentum * layer.bias_momentums) - (self.current_learning_rate * layer.gradient_biases)
            layer.bias_momentums = bias_updates

        else:  #Default update (Vanilla SGD updates and normalisation) if momentum isn't being used. 
            weight_updates = -self.current_learning_rate * layer.gradient_weights
            bias_updates = -self.current_learning_rate * layer.gradient_biases

        #Apply updates
        layer.weights += weight_updates
        layer.biases += bias_updates

        #In each epoch (or step), the weights will be adjusted by subtracting the learning rate multiplied by the gradient weights.
        #In each epoch (or step), the biases will be adjusted by subtracting the learning rate multiplied by the gradient biases.

    def post_update_parameters(self):
        """
        Increment the iteration count after parameters have been updated.

        Keeps track of the number of 'steps' performed by the optimiser.
        """
        self.iterations += 1

class AdagradOptimiser:
    """
    Adagrad optimiser class. One of the optimisers that could be implemented during training.

    The Adagrad optimiser adapts the learning rate based on history of gradient information. This factors in larger steps for infrequent features and smaller steps for frequent features.
    """

    def __init__(
        self, learning_rate: float = 1.0, decay: float = 0.0
    ):  # Setting a learning rate of 1 as the default for this optimiser, which will be decayed over time according to the decay rate.
        """
        Initialises the Adagrad optimise with initial/default values for learning rate and decay rate.

        Args:
        learning_rate (float): The initial learning rate, to be decayed over time by the decay rate. Set to 1 to begin with.
        decay (float): The decay rate, which determines the rate at which the learning_rate will be decreased (i.e. the rate at which the 'steps' will be made smaller). Set to 0 to begin with.
        """
        self.learning_rate = learning_rate  # This is the initial learning rate.
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0  # This is the number of 'steps' (as it's known in training), that the model has gone through.
        self.EPSILON = 1e-7

    def pre_update_parameters(self):
        """
        Updates the current learning rate based on the decay, if applicable.

        This method adjusts the learning rate befor the parameter updates occur according to the relevant Adagrad equations and the decay parameter.
        """
        if self.decay:  # This will only be performed if the decay rate is not 0.
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )  # If there is a decay rate other than 0, this will update the self.current_learning_rate attribute using the relevant formula. The formula essentially multiplies the initial learning rate by 1/(the decay rate * the number of steps). The added 1 makes sure that the algorithm never raises the learning rate. For example, in the first step, we must divide 1 by the learning rate (0.001 for example), which would result in a current learning rate of 1000, which was not the intended result. The added 1 ensures that the result is a fraction of the starting learning rate.

    def update_parameters(self, layer: Layer):
        """
        Updates the weights and biases of the given layer using the relevant Adagrad equations.

        This method modifies the layer's weight and bias caches, applying Adagrad updates based on the current gradients of the layer.

        Args:
        layer (Layer): The layer object whose parameters are to be updated.
        """
        if not hasattr(
            layer, "weight_cache"
        ):  # This will only perform on the first loop, when there are no weight caches or bias caches.
            layer.weight_cache = numpy.zeros_like(layer.weights)
            layer.bias_cache = numpy.zeros_like(layer.biases)

        # Performs correct cache update for the Adagrad optimiser.
        layer.weight_cache += layer.gradient_weights**2
        layer.bias_cache += layer.gradient_biases**2

        # Vanilla SGD parameter updates and normalisation.
        layer.weights += (
            -self.current_learning_rate
            * layer.gradient_weights
            / (numpy.sqrt(layer.weight_cache) + self.EPSILON)
        )
        layer.biases += (
            -self.current_learning_rate
            * layer.gradient_biases
            / (numpy.sqrt(layer.bias_cache) + self.EPSILON)
        )

    def post_update_parameters(self):
        """
        Increments the iteration count after parameters have been updated. This marks the next step.

        This method is called after the updates to keep track of the number of 'steps' performed by the optimiser.
        """
        self.iterations += 1


class RMSPropOptimiser:
    """
    RMSProp optimiser class. One of the optimisers that could be implemented during training.

    The RMSProp optimiser calculates an adaptive learning rate per parameter - like Adagrad - just in a different way.
    """

    def __init__(self, learning_rate: float = 0.001, decay: float = 0.0):
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
        self.RHO = 0.9  # The main additional parameter between Adagrad and RMSProp is rho. Rho is the cache memory decay rate.

    def pre_update_parameters(self):
        """
        Updates the current learning rate based on decay, if applicable.

        This method adjusts the learning rate before the parameter updates occur according to the relevant RMSProp equations and the decay parameter.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + (self.decay * self.iterations))
            )  # If decay is being implemented, like Adagrad, RMSProp calculates an adaptive learning rate PER parameter, just in a different way to Adagrad. Whilst AdaGrad calculates the cache as cache += gradient ** 2, RMSProp calculates the cache as cache = rho * cache + (1 - rho) * gradient ** 2. Instead of continually adding squared gradients to a cache (like in Adagrad), it uses a moving average of the cache. Each update to the cache retains a part of the cache and updates it with a fraction of the new, squared, gradients.

    def update_parameters(self, layer: numpy.ndarray):
        """
        Updates the weights and biases of the given layer using the relevant RMSProp equations.

        This method modifies the layer's weight and bias caches, applying RMSProp updates based on the current gradients of the layer.

        Keyword arguments:
        layer (Layer): The layer object whose parameters are to be updated.
        """
        if not hasattr(
            layer, "weight_cache"
        ):  # Like in the Adagrad optimiser, if the layer does not contain cache arrays (checking for weight cache is the same as checking for bias cache, having one means you have both, not having one means you have neither). If they do not exist, create them and fill them with zeros.
            layer.weight_cache = numpy.zeros_like(layer.weights)
            layer.bias_cache = numpy.zeros_like(layer.biases)

        # Performs correct cache update for the RMSProp optimiser.
        layer.weight_cache = (self.RHO * layer.weight_cache) + (
            (1 - self.RHO) * layer.gradient_weights**2
        )
        layer.bias_cache = (self.RHO * layer.bias_cache) + (
            (1 - self.RHO) * layer.gradient_biases**2
        )

        # Vanilla SGD parameter updates and normalisation.
        layer.weights += (
            -self.current_learning_rate
            * layer.gradient_weights
            / (numpy.sqrt(layer.weight_cache) + self.EPSILON)
        )
        layer.biases += (
            -self.current_learning_rate
            * layer.gradient_biases
            / (numpy.sqrt(layer.bias_cache) + self.EPSILON)
        )

    def post_update_parameters(self):
        """
        Increments the iteration count after parameters have been updated. This marks the next step.

        This method is called after the updates to keep track of the number of 'steps' performed by the optimiser.
        """
        self.iterations += 1
        

class AdamOptimiser:
    """
    Adam (Adaptive Momentum) optimiser. One of the optimisers that could be implemented during training.

    The Adam optimiser applies an adaptive learning rate, as in RMSProp, and momentum, as in SGD.
    """
    def __init__(
        self, 
        learning_rate: float = 0.001, 
        decay: float = 0.0
        ):
        """
        Initialise Adam optimiser parameters.

        Args:
        learning_rate: The initial learnig rate. Set to 0.001 to begin with.
        decay: The decay rate for the learning rate. set to 0.0 to begin with.
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
        Update current learning rate according to decay, if applicable.

        Adjusts the learning rate before the parameter updates occur according to the relevant Adam equations and the decay parameter.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_parameters(self, layer: Layer):
        """
        Update the weights and biases of the layer using Adam algorithm.

        Args:
        layer: The layer object whose parameters are to be updated.
        """
        if not hasattr(layer, "weight_cache"):  #Only perform on the first loop, when there are no weight caches or bias caches.
            layer.weight_momentums = numpy.zeros_like(layer.weights)
            layer.weight_cache = numpy.zeros_like(layer.weights)
            layer.bias_momentums = numpy.zeros_like(layer.biases)
            layer.bias_cache = numpy.zeros_like(layer.biases)

        #Update the momentum with current gradients.
        layer.weight_momentums = (self.BETA_1 * layer.weight_momentums) + ((1 - self.BETA_1) * layer.gradient_weights)
        layer.bias_momentums = (self.BETA_1 * layer.bias_momentums) + ((1 - self.BETA_1) * layer.gradient_biases)

        #Get corrected momentums (iterations start at 0, therefore, plus 1).
        weight_momentums_corrected = layer.weight_momentums / (1 - (self.BETA_1 ** (self.iterations + 1)))
        bias_momentums_corrected = layer.bias_momentums / (1 - (self.BETA_1 ** (self.iterations + 1)))

        #Update cache with squared gradients.
        layer.weight_cache = (self.BETA_2 * layer.weight_cache) + ((1 - self.BETA_2) * layer.gradient_weights**2)
        layer.bias_cache = (self.BETA_2 * layer.bias_cache) + ((1 - self.BETA_2) * layer.gradient_biases**2)

        #Compute bias-corrected estimates. 
        weight_cache_corrected = layer.weight_cache / (1 - (self.BETA_2 ** (self.iterations + 1)))
        bias_cache_corrected = layer.bias_cache / (1 - (self.BETA_2 ** (self.iterations + 1)))

        #Vanilla SGD parameter updates and normalisation with adaptive learning rates.
        layer.weights += (-self.current_learning_rate * weight_momentums_corrected / (numpy.sqrt(weight_cache_corrected) + self.EPSILON))
        layer.biases += (-self.current_learning_rate * bias_momentums_corrected / (numpy.sqrt(bias_cache_corrected) + self.EPSILON))

    def post_update_parameters(self):
        """
        Increment the iteration count after parameters have been updated. 

        Keeps track of the number of 'steps' performed by the optimiser.
        """
        self.iterations += 1


class Accuracy:
    """Base class for accuracy calculations."""
    
    def __init__(self): 
        """
        Intialises key values for Accuracy class.
        """
        self.accumulated_sum = 0.0
        self.accumulated_count = 0

    def calculate(self, predicted_probabilities: numpy.ndarray, true_labels: numpy.ndarray) -> float:
        """
        Calculate accuracy for a single batch. 
        
        Args: 
        predicted_probabilities: Model output predicted probabilities, shaped as (number of samples, number of classes).
        true_labels: Ground truth labels, either sparse or one-hot encoded.
        
        Returns: 
        float: Accuracy percentage (0-1). 
        """
        comparisons = self.compare(predicted_probabilities, true_labels)

        accuracy = numpy.mean(comparisons)

        self.accumulated_sum += numpy.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self) -> float: 
        """
        Calculate mean accuracy from accumulated values. 
        
        Returns: 
        float: Mean accuracy percentage (0-1). 
        """
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy

    def reset_epoch(self):
        """Reset accumulated accuracy values for each new training epoch."""
        self.accumulated_sum = 0
        self.accumulated_count = 0


class RegressionAccuracy(Accuracy):
    """Accuracy metric for regression tasks using threshold-based comparison."""
    
    def __init__(self):
        """Initialises key values from parent class Accuracy."""
        super().__init__()
        self.precision = None

    def initialise(self, true_labels: numpy.ndarray, reinit: bool = False):  
        """
        Initialise precision threshold based on target data distribution. 
        
        Args: 
        true_labels: Ground truth values used to calculate precision threshold. 
        reinit: Boolean flag that forces reinitialisation even if already initialised. 
        """
        if self.precision is None or reinit: #For regression, the init method will calculate an accuracy precision. Initialisation won't recalculate precision unless forced to do so by setting the reinit parameter to True. This allows for multiple use-cases, including setting self.precision independently, calling initialise whenever needed (e..g from outside of the model), and even calling it multiple times.
            self.precision = numpy.std(true_labels) / 250 #numpy.std() is a function that helps compute the standard deviation. 

    def compare(self, predicted_probabilities: numpy.ndarray, true_labels: numpy.ndarray) -> numpy.ndarray:
        """
        Compare predictions to true labels within precision threshold. 
        
        Args: 
        predicted_probabilities: Model output predicted probabilities, shaped as (number of samples, number of classes).
        true_labels: Ground truth labels, either sparse or one-hot encoded.
        
        Returns: 
        numpy.ndarray: Boolean array of predictions within precision threshold. 
        """
        return numpy.absolute(predicted_probabilities - true_labels) < self.precision
        

class CategoricalAccuracy(Accuracy):
    """Categorical accuracy for classification tasks."""
    
    def __init__(self, binary: bool = False):
        super().__init__()
        self.binary = binary

    def initialise(self, true_labels: numpy.ndarray):
        """
        Initialise accuracy metric (not applicable for categorical accuracy). 
        
        Args: 
        true_labels: Ground truth values, either sparse or one-hot encoded. 
        """
        pass  #No initialisation needs to be performed here, but empty method must exist as it's going to be called from train method of Model class and would otherwise cause error.

    def compare(self, predicted_probabilities: numpy.ndarray, true_labels: numpy.ndarray): 
        """
        Compare predictions to true labels. 
        
        Args: 
        predicted_probabilities: Model output predicted probabilities, shaped as (number of samples, number of classes).
        true_labels: Ground truth labels, either sparse or one-hot encoded.
        
        Returns: 
        numpy.ndarray: Boolean array of correct predictions. 
        """
        if not self.binary and len(true_labels.shape) == 2:
            true_labels = numpy.argmax(true_labels, axis=1)
        
        return predicted_probabilities == true_labels


class Model:
    """Main neural network model class managing layers, training, and evaluation."""

    def __init__(self):
        """
        Initialise an empty model with empty layers and components. 
        """
        self.layers = []
        self.trainable_layers = []
        self.softmax_classifier_output = None

    def add_layer(self, layer: "Layer | DropoutLayer | ReluActivation | LinearActivation | SigmoidActivation | SoftmaxActivation"):
        """
        Add a layer to the neural network. 
        
        Args: 
        layer: Layer to add (Dense, Dropout, Acitvation, etc.).
        """
        self.layers.append(layer)

    def set(
        self,
        loss: Loss = None,
        optimiser: "StochasticGradientDescentOptimiser | AdagradOptimiser | AdamOptimiser" = None,
        accuracy: Accuracy = None
        ):
        """
        Configure model components. 
        
        Args: 
        loss: Loss function to use for training. 
        optimiser: Optimisation algorithm to use.
        accuracy: Accuracy metric to track.
        """
        if loss is not None:
            self.loss = loss
        if optimiser is not None:
            self.optimiser = optimiser
        if accuracy is not None:
            self.accuracy = accuracy

    def finalise(self):
        """
        Finalise model architecture by connecting layers and initialising components. 
        
        Performs setup operations: 
        1. Connects layers in sequential order
        2. Identifies trainable layers with parameters
        3. Handles special case for softmax-crossentropy optimisation. 
        4. Links loss function to trainable layers for regularisation. 
        """
        #Create and connect input layer. 
        self.input_layer = InputLayer()
        layer_count = len(self.layers)

        #Connect layers in sequence. 
        for i in range(layer_count):
            
            if i == 0:
                self.layers[i].prev = (self.input_layer)  # .prev and .next are dynamically assigned attributes to connect layers, which is part of the reason a separate - although small - class was made for input layer.
                self.layers[i].next = self.layers[i + 1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"): 
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:  #Check if the model is being loaded with pre-set parameters, meaning it doesn't need to be trained, hence not needing an optimiser.
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], SoftmaxActivation) and isinstance(self.loss, CategoricalCrossEntropyLoss):  #Check to see if the softmax activation AND categorical cross entropy classes are being used. If so, the softmax classifier previously developed and described can be used to combine the two.
            self.softmax_classifier_output = SoftmaxClassifier()

    def train(
        self,
        training_data: numpy.ndarray,
        training_labels: numpy.ndarray,
        epochs: int = 1,
        batch_size: int = None,
        print_step: int = 1,
        validation_data: tuple = None,
        ):
        """
        Train model on labelled data using configured components. 
        
        Implements the following training process: 
        1. Forward propagation through network
        2. Loss calculation with regularisation
        3. Backward propagation for gradients 
        4. Parameter updates via optimiser
        5. (Optional) validation
        
        Args: 
        training_data: Input features array, shaped as (number of samples, number of neurons/features).
        training_labels: Target values array of shape (number of samples, number of neurons/features). 
        epochs: Number of complete passes through training data.
        batch_size: Number of samples per gradient update (0 = full batch). 
        print_step: Report every [print_interval] batches.
        validation_data: Optional tuple (validation_data, validation_labels) for each epoch validation. 
        """
        
        if None in (self.loss, self.optimiser, self.accuracy): 
            raise RuntimeError("Model must be configured with loss, optimsier, and accuracy before training")

        self.accuracy.initialise(training_labels)

        train_steps = self.calculate_batch_steps(training_data, batch_size)

        for epoch in range(1, epochs + 1):

            print(f"epoch: {epoch}")

            self.reset_tracking_metrics()

            for step in range(train_steps): 
                
                batch_training_data, batch_labels = self.get_batch_data(training_data, training_labels, step, batch_size)
        
                predictions = self.forward_pass(batch_training_data, training=True)

                data_loss, regularisation_loss = self.loss.calculate(predictions, batch_labels, regularisation=True)

                loss = data_loss + regularisation_loss

                class_predictions = self.output_layer_activation.predictions(predictions)
                batch_accuracy = self.accuracy.calculate(class_predictions, batch_labels)

                self.backward_pass(predictions, batch_labels)

                self.execute_parameter_updates()

                if not step % print_step == 0 or step == train_steps - 1:
                    self.print_training_update(step, train_steps, batch_accuracy, loss, data_loss, regularisation_loss)

            epoch_data_loss, epoch_regularisation_loss = (self.loss.calculate_accumulated(regularisation=True))

            epoch_loss = epoch_data_loss + epoch_regularisation_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            
            print(f"Training Summary |"
                  f"Accuracy: {epoch_accuracy:.3f} "
                  f"| Loss: {epoch_loss:.3f}"
                  f"(Data: {epoch_data_loss:.3f}, Regularisation: {epoch_regularisation_loss:.3f})")

            if validation_data:
                self.evaluate(*validation_data, batch_size=batch_size)  #*validation_data unpacks the validation_data list into singular values (e.g. tuple = (1,2) *tuple = 1 2)

    def evaluate(self,
                validation_data: numpy.ndarray,
                validation_labels: numpy.ndarray,
                batch_size: int = None
                ):
        """
        Evaluate model performance on validation dataset.
        
        Args: 
        validation_data: Input features array, shaped as (number of samples, number of neurons/features).
        validation_labels: Target values array of shape (number of samples, number of neurons/features). 
        batch_size: Number of samples per evaluation batch (0 = full batch).  
        """
        validation_steps = self.calculate_batch_steps(validation_data, batch_size)
        
        self.reset_tracking_metrics()

        for step in range(validation_steps):
            batch_data, batch_labels = self.get_batch_data(validation_data, validation_labels, step, batch_size)
            
            predictions = self.forward_pass(batch_data, training=False)

            self.loss.calculate(predictions, batch_labels)

            class_predictions = self.output_layer_activation.predictions(predictions)

            self.accuracy.calculate(class_predictions, batch_labels)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f"Validation Summary |"
              f"Accuracy: {validation_accuracy:.3f}"
              f"| Loss: {validation_loss:.3f}")
                  
        
    def predict(self, input_data: numpy.ndarray, batch_size: int = None) -> numpy.ndarray:
        """
        Generate model predictions for input data. 
        
        Args: 
        input_data: Input array, shaped as (number of samples, number of neurons/features).
        batch_size: Number of samples per prediction batch (0 = full batch).
        
        Returns: 
        numpy.ndarray: Model predictions array, shaped as either (number of samples, number of classes) probability distribution (classification) or (number of samples, output dimensions) continuous values (regression). 
        """
        prediction_steps = self.calculate_batch_steps(input_data, batch_size)
        predictions = []
        
        for step in range(prediction_steps):
            if batch_size is None:
                batch_data = input_data

            else:
                batch_data = input_data[step * batch_size : (step + 1) * batch_size]

            batch_predictions = self.forward_pass(batch_data, training=False)
            predictions.append(batch_predictions)

        return numpy.vstack(predictions) #numpy.vstack is used to stack arrays vertically. It concatenates arrays vertically, equivalent to np.concatenate() with axis = 0. 
    
    def forward_pass(self, input_data: numpy.ndarray, training: bool) -> numpy.ndarray:
        """
        Perform forward propagation through the network. 
        
        Args: 
        input_data: Input array, shaped as (number of samples, number of neurons/features).
        training: Boolean flag indicating training mode. 
        
        Returns: 
        numpy.ndarray: Network output after passing through all layers. 
        """
        self.input_layer.forward_pass(input_data, training)

        for layer in self.layers:
            layer.forward_pass(layer.prev.output, training)

        return layer.output

    def backward_pass(self, predictions: numpy.ndarray, true_labels: numpy.ndarray):
        """
        Perform backward propagation through the network. 
        
        Computes gradients using either optimised softmax-crossentropy combined implementation or standard loss function backpropagation. 
        
        Args: 
        predictions: Network outputs from forward pass.
        true_labels: Ground truth labels for gradient calculation. 
        """

        if self.softmax_classifier_output:  # Checks if the softmax classifier is being used in the model

            self.softmax_classifier_output.backward_pass(predictions, true_labels)

            self.layers[-1].gradient_inputs = self.softmax_classifier_output.gradient_inputs

            for layer in reversed(self.layers[:-1]):
                layer.backward_pass(layer.next.gradient_inputs)

        else:

            self.loss.backward_pass(predictions, true_labels)

            for layer in reversed(self.layers):
                layer.backward_pass(layer.next.gradient_inputs)

    def save(self, file_path: str):
        """
        Serialise entire model object to storage. 
        
        Handles Python object serialisation while removing temporary attributes. 
        
        Args: 
        file_path: Path to save model (should have .model extension). 
        """
        model = copy.deepcopy(self)

        model.loss.reset_epoch()
        model.accuracy.reset_epoch()

        model.input_layer.__dict__.pop("output", None)
        model.loss.__dict__.pop("gradient_inputs", None)

        for layer in model.layers:
            for property in ["inputs","outputs","gradient_inputs","gradient_weights","gradient_biases"]:
                layer.__dict__.pop(property, None)

        with open(file_path, "wb") as file:
            pickle.dump(model, file)

    @staticmethod #This must be a static method as you should be able to use it without instantiating an object of the Model class. 
    def load(file_path: str) -> "Model": 
        """
        Deserialise model from storage. 
        
        Args: 
        file_path: Path to saved model file. 
        
        Returns: 
        "Model": Reconstructed Model instance. 
        """
        with open(file_path, "rb") as file:
            model = pickle.load(file)

        return model
    
    def calculate_batch_steps(self, data: numpy.ndarray, batch_size: int = None) -> int: 
        """
        Calculate number of batches for given data size. 
        
        Args: 
        data: Full dataset array, shaped as (number of samples, number of neurons/features).
        batch_size: Samples per batch.
        
        Returns: 
        int: Number of batches needed to process all data.
        """
        if batch_size is None: 
            return 1
        
        steps = len(data) // batch_size
        
        if steps * batch_size < len(data): 
            steps += 1
            
        return steps 
    
    def get_batch_data(self,
                       data: numpy.ndarray,
                       labels: numpy.ndarray,
                       step: int,
                       batch_size: int = None
                       ) -> tuple:
        """
        Extract batch from dataset.
        
        Args: 
        data: Full dataset array, shaped as (number of samples, number of neurons/features).
        labels: Corresponding labels array.
        step: Current batch index. 
        batch_size: Samples per batch.
        
        Returns: 
        tuple: Shaped as (batch_data, batch_labels).
        """
        if batch_size is None: 
            return data, labels
        
        start = step * batch_size
        end = start + batch_size
        
        return data[start:end], labels[start:end]
    
    def reset_tracking_metrics(self): 
        """Reset accumulated loss and accuracy metrics for new epoch."""
        self.loss.reset_epoch()
        self.accuracy.reset_epoch()
        
    
    def execute_parameter_updates(self): 
        """
        Execute parameter updates using configured optimiser. 
        """
        self.optimiser.pre_update_parameters()

        for layer in self.trainable_layers:
            self.optimiser.update_parameters(layer)

        self.optimiser.post_update_parameters()
        
    def print_training_update(
        self,
        step: int,
        total_steps: int,
        accuracy: float,
        total_loss: float,
        data_loss: float,
        regularisation_loss: float
        ):
        """
        Format and print training progress update.
        
        Args: 
        step: Current batch index. 
        total_steps: Total number of steps to be performed during training. 
        accuracy: Calculated accuracy at that stage in training.
        total_loss: Total loss at that stage in training. 
        data_loss: A component of total loss.
        regularisation_loss: An optional component of total loss. 
        """
        print(f"Step {step+1:>{len(str(total_steps))}}/{total_steps} | "
              f"Accuracy: {accuracy:.3f} | "
              f"Loss: {total_loss:.3f} (Data: {data_loss:.3f}, Regularisation: {regularisation_loss:.3f}) | "
              f"Learning Rate: {self.optimiser.current_learning_rate:.6f}")
        

