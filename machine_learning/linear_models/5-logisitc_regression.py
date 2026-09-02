#!/usr/bin/env python3
"""
A function to create a logistic regression model which performs
binary classification by fitting a logistic function.

Arguments:
random_state: An integer used to set the random seed for reproducibility.

Returns:
model: An untrained LogisticRegression instance.
"""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """
    Create a logistic regression model, which performs binary classification
    by fitting a logistic function
    """

    return linear_model.LogisticRegression(random_state=random_state)
