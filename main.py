import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sys


'''
major inspirations from:
https://enlight.nyc/projects/neural-network/
'''

class nn:
    def __init__(self, layers = 1):
        #set system recursion depth to wayyyyy higher than default
        sys.setrecursionlimit(10**6)
        #create two empty variables that will by numpy arrays for our data
        self.x_data = None
        self.y_data = None
        #our weights, acts as our brains neurons
        self.W1 = None # (3x2) weight matrix from input to hidden layer
        self.W2 = None # (3x1) weight matrix from hidden to output layer

        #one bias per layer

    #activation function
    def sigmoid(self, x):
        return 1/1+np.exp(-x)
    #derivative of our activation function
    def sigmoid_dir(self, x):
        return x * (1 - x)
    #make predictions using the nn
    def predict(self, inputs):
        self.z = np.dot(inputs, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        prediction = self.sigmoid(self.z3) # final activation function
        return prediction

    def backprop(self, X, y, o):
        # backward propagate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoid_dir(o) # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoid_dir(self.z2) # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

    def train(self, data, labels, epochs):
        #init weights
        hiddenSize = 3
        data_shape = np.shape(data)[1]
        #print(data_shape)
        self.W1 = np.random.randn(data_shape, hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(hiddenSize, 1) # (3x1) weight matrix from hidden to output layer

        '''
        this is the main training loop where a for loop would be if this was
        made by any other normal intelligent human being

        see, the thing is, i am neither intelligent nor normal therefor i'm
        using recursion

        there is no speed improvement, infact it may actually sometimes be slower
        this is simply to be a big flex on my part
        '''
        def recursion_loop(epochs):
            #sexy loading bar thing lol
            print("#", end = "")
            '''
            ******* TRAIN NN *******
            '''
            o = self.predict(data)
            self.backprop(data, labels, o)
            #if the number of epochs is above 0, continue recursion!
            #if not then just stop
            #increment the reaming epochs down by 1
            '''
            EXIT OR CONTINUE RECURSION
            '''
            if epochs > 0:
                recursion_loop(epochs-1)

        #call recursion_loop
        recursion_loop(epochs)

def example():
    #data
    net = nn()
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    net.train(training_inputs, training_outputs, 1000)
    print("")
    print(net.predict(np.array([0,0,1])))
    for i in range(10):
        print("_ _ _ ", end="")
    print("")
    print(net.W1)
    print(net.W2)

if __name__ == '__main__':
    example()
