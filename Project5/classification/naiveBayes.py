# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        # Calculate P(Y)
        p = util.Counter()
        for y in trainingLabels:
            p[y] += 1
        p.normalize()
        self.p = p

        # initialize counts, totals of each feature and value
        # each count is dict of {feature: {feature value: Counter(y value)}}
        counts = dict()
        totals = dict()
        for feature in self.features:
            counts[feature] = {0: util.Counter(), 1: util.Counter()}
            totals[feature] = util.Counter()
        
        for x, y in zip(trainingData, trainingLabels):
            for feature, value in x.items():
                counts[feature][value][y] += 1.
                totals[feature][y] += 1.

        best_score = float('-inf')
        best_k = 0
        best_conditional_probs = dict()
        for k in kgrid:
            # initialize conditional probabilities we will be smoothing
            conditional_probs = dict()
            for feature in self.features:
                conditional_probs[feature] = {0: util.Counter(), 1: util.Counter()}

            # smooth
            for feature in self.features:
                for y in self.legalLabels:
                    conditional_probs[feature][0][y] = (counts[feature][0][y] + k) / (totals[feature][y] + k)
                    conditional_probs[feature][1][y] = (counts[feature][1][y] + k) / (totals[feature][y] + k)

            # validate
            self.conditional_probs = conditional_probs
            output = self.classify(validationData)
            
            correct_count = 0
            for y, y_star in zip(output, validationLabels):
                if y == y_star:
                    correct_count += 1.
            
            score = correct_count/len(output)

            if score > best_score:
                best_k = k
                best_score = score
                best_conditional_probs = conditional_probs

        self.k = best_k
        self.conditional_probs = best_conditional_probs


    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()
        
        for y in self.legalLabels:
            logJoint[y] = math.log(self.p[y])

            for feature, value in datum.items():
                prob = self.conditional_probs[feature][value][y]
                if prob > 0:
                    logJoint[y] += math.log(self.conditional_probs[feature][value][y])

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        for feature in self.features:
            ratio = self.conditional_probs[feature][1][label1] / self.conditional_probs[feature][1][label2]
            featuresOdds.append((feature, ratio))

        # sort by the ratio
        featuresOdds.sort(key=lambda tup: tup[1])

        return featuresOdds[:100]
