#!/usr/bin/env python3
"""
A function to train a decision tree classifier using Scikit-learn.

Arguments:
clf: A Scikit-learn classifier instance
X: Input features
y: Target labels

Returns:
None
"""
from sklearn import tree


def train_tree(clf, X, y):
    """ A function to train a decision tree classifier """

    return clf.fit(X, y)
