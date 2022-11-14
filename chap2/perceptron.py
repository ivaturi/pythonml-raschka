from utils.vector import dot
import random
import logging
import numpy as np
import pandas 


class Perceptron:

    def __init__(self, ndims = 2, learning_rate = 0.1, bias = 0.1, max_iter = 20):
        """
        Initialize a perceptron of size n
        """
        self.max_iter = max_iter
        self.ndims = ndims
        self.errors = []
        # seed the random number generator (for consistency across trials)
        random.seed(0)
        self.W = [random.random() for _ in range(0, self.ndims)]
        self.learning_rate = learning_rate
        self.bias = bias


    def predict(self, X):
        """
        Compute the output for a given dataset X
        """
        weighted_input = dot(self.W, X) + self.bias
        return self.activation(weighted_input)

    def activation(self, z):
        return 1 if z > 0 else -1

    def fit(self, X, Y):
        for _ in range(0, self.max_iter):
            # reset error at the the beginning of each iteration
            error = 0.0
            for x, expected in zip(X, Y):
                predicted = self.predict(x)
                error += expected - predicted
                update  = self.learning_rate * error
                self.bias = self.bias + update
                self.W = [wi + update*xi for xi, wi in zip(x,self.W)]
            logging.debug("Iter %d, error = %f" % (_, error))
    
    def show(self, X):
        pass

class PerceptronNP:
    """
    Perceptron implementation using numPY, from the book
    """

    def __init__(self, learning_rate = 0.01, max_epochs = 50):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state =1

    def fit(self, X, y):
        """
        Fit training data
        """
        # initialize random number generator
        rgen = np.random.RandomState(self.random_state)
        # draw random numbers from a normal distribution with stddev 0.01
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1  + X.shape[1])
        
        self.errors_ = []

        for _ in range(0, self.max_epochs):
            errors = 0
            for xi, target in zip(X, y):
                # calculate the weight update
                update = self.eta * (target - self.predict[xi])
                # update weights
                self.w_[1:] += update*xi    
                # update bias
                self.w_[0] += update     
                # all non-zero updates are errors
                errors += int(update != 0.0)   
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)