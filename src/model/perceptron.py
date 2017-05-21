# -*- coding: utf-8 -*-
from __future__ import division
import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test,
                                    learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.trainingSet.input = self._augment_data(self.trainingSet.input)
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/100
        #self.weight[0] = 0.0
        print self.trainingSet.input.shape
        #print self.trainingSet.label[0]
    def _augment_data(self, data):
        """ augmentation of first dimension of data as 1
        for bias computation
        """
        num = data.shape[0]
        return np.hstack([np.ones(num).reshape(num, 1), data])
    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        t = np.asarray(self.trainingSet.label)
        ufunc_sigh = np.frompyfunc(Activation.sign, 1, 1)
        for i in xrange(self.epochs):
            if verbose:
                pred = self.evaluate(
                    test=self._augment_data(self.validationSet.input))
                print("Epoch: {0}, Accuracy on validation data: {1:.2f}".format(
                        i, accuracy_score(self.validationSet.label, pred)*100))
            o = ufunc_sigh(np.matmul(
                self.trainingSet.input, self.weight.transpose())).astype(np.int)
            error = t-o
            self.updateWeights(self.trainingSet.input, error)

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Write your code to do the classification on an input image
        return self.fire(testInstance)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self._augment_data(self.testSet.input)
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, input, error):
        # Write your code to update the weights of the perceptron here
        self.weight += self.learningRate * np.matmul(error, input)

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))
