#!/usr/bin/env python3
"""
A function to to generate predictions from a trained tree-based classifier.

Arguments:
clf: A trained Scikit-learn classifier instance
X: Feature matrix (NumPy array or pandas DataFrame)

Returns:
A NumPy array containing the predicted class labels for the input samples.
"""


def generate_predictions(clf, X):
    """
    A function to generate predictions from a trained tree-based classifier
    """

    return clf.predict(X)
