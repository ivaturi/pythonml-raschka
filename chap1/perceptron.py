from utils.vector import dot
import random
import logging

class Perceptron():

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

    def learn(self, X, Y):
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

def test_perceptron():
    # learning data
    X = [
        [1,1],
        [1,2],
        [1,3],
        [2,1],
        [2,2],
        [4,1],
        [4,2],
        [4,3],
        [5,2],
        [5, 3],
        [4,6]
    ]
    Y = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1]

    # perceptron
    p = Perceptron()
    p.learn(X, Y)

    # test data
    X = [
        [2, 5],
        [3,0],
        [10,10],
        [4, 8],
        [1, 8]
        ]

    expected = [-1, 1, 1, 1, -1]
    for x, expected in zip(X, expected):
        y = p.predict(x)
        print("E: %d, P: %d" % (expected, y))