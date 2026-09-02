#!/usr/bin/env python3
"""
A function that creates and returns a Lasso Regression model.

Lasso Regression extends ordinary linear regression by adding
L1 regularization, which helps simplify the model by forcing some coefficients
to zero, enabling automatic feature selection.

Arguments:
random_state: An integer used to set the random seed for reproducibility.

Returns:
model: An untrained Lasso regression model instance.
"""
from sklearn import linear_model


def lasso_regression(random_state):
    """
    A function that creates and returns a Lasso Regression model.
    """

    return linear_model.Lasso(random_state=random_state)
