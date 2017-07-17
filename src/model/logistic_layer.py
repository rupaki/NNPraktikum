
import time

import numpy as np

from util.activation_functions import Activation
#from model.layer import Layer


class LogisticLayer():
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    delta : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, learningRate=0.01, weights=None,
                 activation='sigmoid', isClassifierLayer=True):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.activation_derivative = Activation.getDerivative(
                                                        self.activationString)
        self.learningRate = learningRate
        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.input = np.ndarray((nIn+1, 1))
        self.input[0] = 1
        self.output = np.ndarray((nOut, 1))
        self.delta = np.zeros((nOut, 1))

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nIn + 1, nOut))-0.5
        else:
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

    def _augment_data(self, data):
        """ augmentation of first dimension of data as 1
        for bias computation
        """
        #print data.shape
        if len(data.shape) == 1:
            new_data = np.zeros(1+data.shape[0])
            #print new_data.shape
            new_data[0] = 1
            new_data[1:] = data
            return new_data
        else:
            num = data.shape[0]
            return np.hstack([np.ones(num).reshape(num, 1), data])
    def forward(self, input):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        input : ndarray
            #a numpy array (1,nIn + 1) containing the input of the layer
            a numpy array (1,nIn) containing the input of the layer
        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        self.input = self._augment_data(input)
        self.output = self.activation(self._fire(self.input, self.weights))
        return self.output

    def _fire(self, input, weights):
        return np.dot(np.array(input), weights)

    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (back)

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer

        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer
        """

        self.delta = self.activation_derivative(self.output) * nextDerivatives * nextWeights
    def updateWeights(self):
        """
        Update the weights of the layer
        """
        for i in range(self.nOut):
            self.weights[:, i] += self.learningRate*self.delta[i]*self.input
