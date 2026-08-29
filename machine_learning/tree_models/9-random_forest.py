#!/usr/bin/env python3
"""
A function to create a random forest classifier using Scikit-learn.

Arguments:
n_estimators: Number of trees in the forest.
random_state: Seed used by the random number generator for reproducibility.

Returns:
model: A Scikit-learn RandomForestClassifier instance.
"""
from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """ Create a random forest classifier """

    return ensemble.RandomForestClassifier(
            n_estimators,
            random_state=random_state
    )
