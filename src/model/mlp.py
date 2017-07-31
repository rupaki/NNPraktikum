
import numpy as np

from util.loss_functions import CrossEntropyError
from util.loss_functions import DifferentError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification',
                 hiddenActivation='sigmoid', outputActivation='softmax',
                 loss='crossentropy', learningRate=0.01, epochs=50,
                 list_hidden_neurons=[100], num_output=10):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int
        list_hidden_neurons: list

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        self.hiddenActivation = hiddenActivation
        #self.cost = cost
        self.list_hidden_neurons = list_hidden_neurons
        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        self.num_output = num_output
        self.train_label = self.label_to_one_hot(self.trainingSet.label, self.num_output)
        #from IPython import embed; embed()
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        elif loss == 'crossentropy':
            self.loss = CrossEntropyError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        print('Setup network...')
        if self.layers is None:
            self.layers = []
            if len(self.list_hidden_neurons) != 0:
                num_neurons_first_hidden_layer = list_hidden_neurons[0]
                num_neurons_last_hidden_layer = list_hidden_neurons[-1]
                self.layers.append(LogisticLayer(train.input.shape[1],
                                   num_neurons_first_hidden_layer,
                                   learningRate=self.learningRate,
                                   activation = self.hiddenActivation))
                print('Layer 0: {}x{}, active: {}'.format(train.input.shape[1],
                                                             num_neurons_first_hidden_layer,
                                                             self.hiddenActivation))
                for i, num in enumerate(self.list_hidden_neurons[1:]):
                    self.layers.append(LogisticLayer(self.list_hidden_neurons[i],
                                       num,
                                       learningRate=self.learningRate,
                                       activation = self.hiddenActivation))
                    print('Layer {}: {}x{}, active: {}'.format(i+1,
                                                              self.list_hidden_neurons[i],
                                                              num,
                                                              self.hiddenActivation))
                self.layers.append(LogisticLayer(num_neurons_last_hidden_layer,
                                       self.num_output,
                                       learningRate=self.learningRate,
                                       activation = self.outputActivation))
                print('Layer {}: {}x{}, active: {}'.format(len(self.list_hidden_neurons),
                                                            num_neurons_last_hidden_layer,
                                                            self.num_output,
                                                            self.outputActivation))
            else:
                self.layers.append(LogisticLayer(train.input.shape[1],
                                       self.num_output,
                                       learningRate=self.learningRate,
                                       activation = self.outputActivation))
        #from IPython import embed; embed()

    def label_to_one_hot(self, label, num_output):
        zeros = np.zeros((label.shape[0], num_output))
        zeros[np.arange(label.shape[0]), label] = 1.0
        return zeros
    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        last_out = inp
        for l in self.layers:
            last_out = l.forward(last_out)
        return last_out

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        output_layer = self._get_output_layer()
        #from IPython import embed; embed()
        loss_grad = self.loss.calculateDerivative(target, output_layer.output)
        output_layer.computeDerivative(loss_grad, np.array(1.0))
        #output_layer.delta = target - output_layer.output
        next_derivatvie = output_layer.delta
        next_weights = output_layer.weights
        for i in reversed(range(len(self.layers)-1)):
            current_layer = self._get_layer(i)
            current_layer.computeDerivative(next_derivatvie, next_weights[1:].T)
            next_derivatvie = current_layer.delta
            next_weights = current_layer.weights


    def _update_weights(self):
        """
        Update the weights of the layers by propagating back the error
        """
        for l in self.layers:
            l.updateWeights()

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        learned = False
        iteration = 0
        cog_loss = DifferentError()
        while not learned:
            grad = 0
            totalError = 0
            for input, label in zip(self.trainingSet.input,
                                    self.train_label):

                self._feed_forward(input)
                self._compute_error(label)
                self._update_weights()
                # compute recognizing error, not BCE
                predictedLabel = self.classify(input)
                error = cog_loss.calculateError(np.argmax(label), predictedLabel)
                totalError += error

            #self.updateWeights(grad)
            totalError = abs(totalError)

            iteration += 1

            if verbose:
                print("Epoch: {}; Error: {}".format(iteration, totalError))
                accuracy = accuracy_score(self.validationSet.label,
                          self.evaluate(self.validationSet))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

            if iteration >= self.epochs:
                # stop criteria is reached
                learned = True



    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        return np.argmax(self._feed_forward(test_instance))


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
