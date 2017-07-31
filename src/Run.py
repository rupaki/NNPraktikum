#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron
from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot
import matplotlib.pyplot as plt
from matplotlib import gridspec

def main():
    # data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
    #                                                 oneHot=True)
    data_all = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                        oneHot=False)
    #from IPython import embed; embed()
    # myStupidClassifier = StupidRecognizer(data.trainingSet,
    #                                       data.validationSet,
    #                                       data.testSet)
    #
    # myPerceptronClassifier = Perceptron(data.trainingSet,
    #                                     data.validationSet,
    #                                     data.testSet,
    #                                     learningRate=0.005,
    #                                     epochs=30)
    #myLRClassifier = LogisticRegression(data.trainingSet,
                                        #data.validationSet,
                                        #data.testSet,
                                        #learningRate=0.005,
                                        #epochs=30)
    archis = [[128], [64, 64], [128, 64]]
    active_funs = ['sigmoid', 'relu']
    gs = gridspec.GridSpec(len(active_funs), len(archis))
    params = ((x, y) for x in archis for y in active_funs)
    i = 0
    fig = plt.figure()
    for ar, ha in params:
        r = i % len(active_funs)
        c = (i - r) // len(active_funs)
        ax = fig.add_subplot(gs[r, c])
        myMLPlassifier = MultilayerPerceptron(data_all.trainingSet,
                                            data_all.validationSet,
                                            data_all.testSet,
                                            learningRate=0.005,
                                            epochs=30,
                                            outputActivation='softmax',
                                            hiddenActivation=ha,
                                            list_hidden_neurons=ar,
                                            num_output=10)
        # Report the result #
        print("=========================")
        evaluator = Evaluator()

        # Train the classifiers
        print("=========================")
        print("Training..")

        # print("\nStupid Classifier has been training..")
        # myStupidClassifier.train()
        # print("Done..")
        #
        # print("\nPerceptron has been training..")
        # myPerceptronClassifier.train()
        # print("Done..")

        # print("\nLogistic Regression has been training..")
        # myLRClassifier.train()
        # print("Done..")

        print("\nMLP has been training..")
        myMLPlassifier.train()
        print("Done..")

        # Do the recognizer
        # Explicitly specify the test set to be evaluated
        #stupidPred = myStupidClassifier.evaluate()
        #perceptronPred = myPerceptronClassifier.evaluate()
        #lrPred = myLRClassifier.evaluate()
        mlpPred = myMLPlassifier.evaluate()

        # Report the result
        print("=========================")
        evaluator = Evaluator()

        # print("Result of the stupid recognizer:")
        # #evaluator.printComparison(data.testSet, stupidPred)
        # evaluator.printAccuracy(data.testSet, stupidPred)
        #
        # print("\nResult of the Perceptron recognizer:")
        # #evaluator.printComparison(data.testSet, perceptronPred)
        #evaluator.printAccuracy(data.testSet, perceptronPred)
        #
        # print("\nResult of the Logistic Regression recognizer:")
        # #evaluator.printComparison(data.testSet, lrPred)
        # evaluator.printAccuracy(data.testSet, lrPred)

        print("\nResult of the  MLP recognizer:")
        #evaluator.printComparison(data.testSet, lrPred)
        evaluator.printAccuracy(data_all.testSet, mlpPred)
        # Draw
        plot = PerformancePlot("MLP val.,hidden_layers:{}, {}".format(ar, ha))
        plot.draw_performance_epoch(ax,
                                    myMLPlassifier.performances,
                                    myMLPlassifier.epochs)
        i += 1
    #plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()
