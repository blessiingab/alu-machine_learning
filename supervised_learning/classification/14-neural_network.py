#!/usr/bin/env python3
""" Neural Network
"""

import numpy as np


class NeuralNetwork:
    """ Class that defines a neural network with one hidden layer
        performing binary classification.
    """

    def __init__(self, nx, nodes):
        """ Instantiation function

        Args:
            nx (int): size of the input layer
            nodes (int): number of nodes in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Return weights vector for hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Return bias for hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Return activated output for hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Return weights vector for output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Return bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Return activated output for the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): input data with shape (nx, m)

        Returns:
            tuple: activated hidden layer and output layer
        """
        z = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z))

        z = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): correct labels
            A (numpy.ndarray): activated output

        Returns:
            float: cost of the model
        """
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return np.mean(loss)

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions

        Args:
            X (numpy.ndarray): input data
            Y (numpy.ndarray): correct labels

        Returns:
            tuple: predictions and cost
        """
        self.forward_prop(X)
        return (
            np.where(self.__A2 >= 0.5, 1, 0),
            self.cost(Y, self.__A2)
        )

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent

        Args:
            X (numpy.ndarray): input data
            Y (numpy.ndarray): correct labels
            A1 (numpy.ndarray): hidden layer output
            A2 (numpy.ndarray): output layer output
            alpha (float): learning rate
        """
        m = Y.shape[1]

        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.mean(dz2, axis=1, keepdims=True)

        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.mean(dz1, axis=1, keepdims=True)

        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network

        Args:
            X (numpy.ndarray): input data
            Y (numpy.ndarray): correct labels
            iterations (int): number of training iterations
            alpha (float): learning rate

        Returns:
            tuple: predictions and cost after training
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')

        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
