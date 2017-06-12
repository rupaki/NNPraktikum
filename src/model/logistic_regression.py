# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
from sklearn.metrics import accuracy_score
from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        self.layer = LogisticLayer(self.trainingSet.input.shape[1], 1, learningRate=learningRate)
        # Initialize the weight vector with small values

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for i in xrange(self.epochs):
            if verbose:
                pred = self.evaluate(
                    test=self.validationSet.input)
                print("Epoch: {0}, Accuracy on validation data: {1:.2f}".format(
                        i, accuracy_score(self.validationSet.label, pred)*100))
            self.step_once()
    def step_once(self):
        for da, la in zip(self.trainingSet.input, self.trainingSet.label):
            self.layer.forward(da)
            self.layer.computeDerivative(np.asarray(la - self.layer.output), np.array(1.0))
            self.layer.updateWeights()

    def classify(self, testInstance, threshold=0.5):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        o = self.layer.forward(testInstance)
        return o > threshold

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
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))
