# -*- coding: utf-8 -*-

import sys
import logging
import numpy as np
from sklearn.metrics import accuracy_score
from util.activation_functions import Activation
# from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer

from util.loss_functions import *

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
    learningRate : float
    epochs : positive int
    performances: array of floats
    """

    def __init__(self, train, valid, test,
                 learningRate=0.01, epochs=50,
                 loss='bce'):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        # Initialize the weight vector with small values
        #self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])
        self.layer = LogisticLayer(self.trainingSet.input.shape[1], 1, learningRate=learningRate)

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        from util.loss_functions import BinaryCrossEntropyError, DifferentError
        cog_loss = DifferentError()
        loss = BinaryCrossEntropyError()

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        while not learned:
            grad = 0
            totalError = 0
            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):

                output = self.layer.forward(input)
                # compute gradient
                loss_grad = loss.calculateDerivative(label, output)

                self.layer.computeDerivative(loss_grad, np.array(1.0))
                self.layer.updateWeights()
                # compute recognizing error, not BCE
                predictedLabel = self.classify(input)
                error = cog_loss.calculateError(label, predictedLabel)
                totalError += error

            #self.updateWeights(grad)
            totalError = abs(totalError)

            iteration += 1

            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, totalError)


            # Update weights in the online learning fashion
            self.layer.updateWeights(self.learningRate)

    def classify(self, test_instance):
        """Classify a single instance.

        Parameters
        ----------
        test_instance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.layer.forward(testInstance) > 0.5

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
        #import pdb; pdb.set_trace()
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
