#!/usr/bin/env python3
"""
A function that creates and returns a Ridge Regression model.

Ridge Regression extends ordinary linear regression by adding
L2 regularization, which helps stabilize the model by shrinking
large coefficients.

Arguments:
random_state: An integer used to set the random seed for reproducibility.

Returns:
model: An untrained Ridge regression model instance.
"""
from sklearn import linear_model


def ridge_regression(random_state):
    """
    A function that creates and returns a Ridge Regression model.
    """

    return linear_model.Ridge(random_state=random_state)
