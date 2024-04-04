## FULL CODE UP TO BUT NOT INCLUDING OPTIMIZERS.

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
# Dense layer
class Layer_Dense:
    
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):

        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    # Forward pass
    def forward(self, inputs):
        
        # Remember input values
        self.inputs = inputs
        
        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        
    # Backward pass
    def backward(self, dvalues):
      
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
        
# ReLU activation
class Activation_ReLU:
    
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
        
    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let’s make a copy of values first
        self.dinputs = dvalues.copy()
        
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        
# Softmax activation
class Activation_Softmax:
    
    # Forward pass
    def forward(self, inputs):
        
        # Remember input values
        self.inputs = inputs
        
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
        keepdims=True))
        
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
        keepdims=True)
        
        self.output = probabilities
        
    # Backward pass
    def backward(self, dvalues):
        
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        
        # Enumerate outputs and gradients, single output/dvales are a row of 3 items for each row
        # in the list of each sample provided.
        for index, (single_output, single_dvalues) in \
        enumerate(zip(self.output, dvalues)):
            print("single_output", single_output)
            print("single_dvalues", single_dvalues)
    
            # Flatten output array, -1 means however many rows required as number of outputs changes
            # flatten to be a load of rows with 1 item in each, so a column vector.
            single_output = single_output.reshape(-1, 1)
          
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)
            print("jacob matrix,", jacobian_matrix)
            
           
            # Calculate sample-wise gradient
        
            # this basically converts the 3D nature of jacobain matrices to 2D
            # by summing up the partial derivatives wrt to each input and multiplying
            # | 0.4  0.6  -0.2  |
            # | 0.1  0.4   -0.1 |
            # | -5   1.3   2.5 | > example jacobian matrix for some sample of inputs
            # in this case there are 3 inputs so  3 x 3 matrix
            # each row summed up and multiplied by the single_dvalues [3, 7, 0.4] whatever they are
            # creating a row to be passed down as the dinputs for that specific neuron.
            # repeat in the loop for each sample > this is simplified in the combined equation
            # by the associated d value for every input, where each input has a partial derivative for
            # each of the inputs including itself which is the jacobian matrix basically
            # each row becomes a single sample row from a 2D matrix via this dot product as a result.
            self.dinputs[index] = np.dot(jacobian_matrix,
            single_dvalues)
            print(self.dinputs)
            
# Common loss class
class Loss:
    
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        # Return loss
        return data_loss
    
# Cross-entropy loss, notice it inherits class calculate method.
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
    
        # Number of samples in a batch
        samples = len(y_pred)
        
        # Clip data to prevent division by 0, in case some value has a zero predicted truth value
        # which is not impossible 
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Probabilities for target values -
        # only if categorical labels, if 1D then just take the indexes as is from the range of samples
    
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]

        # Mask values - only for one-hot encoded labels
        # one hot matches the shape and has a 1 in the position of truth
        # so if you sum the multiplication of the numpy arrays it just selects the right value
        # as y_true has a single 1 value.
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum( y_pred_clipped * y_true,axis=1 )
        
        # Losses, return array of loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
    
        # Number of samples
        samples = len(dvalues)
        
        # Number of labels in every sample
        # We’ll use the first sample to count them
        labels = len(dvalues[0])
        
        # If labels are sparse, turn them into one-hot vector
        # turn into 2D basically, if 1D then create a diagonal of 1's of length of the labels.
        if len(y_true.shape) == 1: 
            y_true = np.eye(labels)[y_true] # from the diagonal of 1's we go through each one and select the index it should be
            # to see if its the correct class being chosen as the truth value.
    
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
        
    # Forward pass
    def forward(self, inputs, y_true):
        
        # Output layer’s activation function
        self.activation.forward(inputs)
        
        # Set the output
        self.output = self.activation.output
        
        # Calculate and return loss value 
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        
        # Number of samples, length of a 2d array counts rows i.e. samples in this case.
        samples = len(dvalues)
 
        # If labels are one-hot encoded,
        # turn them into discrete values
        # along the rows find the index with the 1 and this becomes the y_true array to
        # index into dinputs rows with to find the prediction to minus the 1 value off.
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)#
        
            
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        
        # Calculate gradient, negate one as the formulae is predicated value at index,sample - truth value
        # and the truth value is just 1 at the place we want to locate.
        self.dinputs[range(samples), y_true] -= 1
        
        # Normalize gradient, so sample size does not affect relative size its just one value between 0 and 1.
        self.dinputs = self.dinputs / samples
        
# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(3, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)

# Let's see output of the first few samples:
print(loss_activation.output[:5])

# Print loss value
print('loss:', loss)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
    
accuracy = np.mean(predictions==y)

# Print accuracy
print('acc:', accuracy)

# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)