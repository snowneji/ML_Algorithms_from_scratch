import numpy as np
import pandas as pd

#Input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
y=np.array([[1],[1],[0]])


#Sigmoid:
def sigmoid(x):
    return(1/(1+np.exp(-x)))

#Derivative of Sigmoid:
def derivatives_sigmoid(x):
    return x*(1-x)


# Initial Weights:
epoch = 5000
lr = 0.01
inputlayear_neurons = X.shape[1] #4
hiddenlayer_neurons = 3
output_neurons = 1

# Initialize weights and bias:
wh = np.random.uniform(size=(inputlayear_neurons,hiddenlayer_neurons)) #4,3
bh = np.random.uniform(size = (1,hiddenlayer_neurons)) # (1,3)_
wout = np.random.uniform(size = (hiddenlayer_neurons,output_neurons)) #(3,1)
bout = np.random.uniform(size = (1,output_neurons))  #(1,1)

error_list = []
for i in range(epoch):
    # Forward Propagation:
    #hidden layer
    hidden_layer_input = np.dot(X,wh) + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    #Output Layer
    output_layer_input = np.dot(hiddenlayer_activations,wout) + bout
    output = sigmoid(output_layer_input)


    # Then do back propagation:


    #Find the derivative:
    slope_output_layer = derivatives_sigmoid(output) # derivative of output layer
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations) # derivative of hidden layer


    # Find Error at output layer:
    E = y-output # error
    print('Error:',np.mean(E))
    error_list.append(np.mean(E))
    # Using derivative to find change factors at output layer:
    d_output = E * slope_output_layer # change factor of output layer




    # Find Error at hidden layer:
    Error_at_hidden_layer = d_output.dot(wout.T)
    # Using derivative to find change factors at hidden layer:
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer # change factor of hidden layer


    #update:
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

error_list = pd.Series(error_list)
error_list.plot()
