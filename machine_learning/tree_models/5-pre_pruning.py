#!/usr/bin/env python3
"""
A function that uses Scikit-learn to perform a Grid Search for the best
pre-pruning hyperparameters for a decision tree classifier.

The search explores the following hyperparameters:
criterion: "gini" or "entropy"
max_depth: integer values in the range [2, 5)
min_samples_leaf: integer values in the range [2, 5)
min_samples_split: integer values in the range [2, 5)

Arguments:
X: Input features
y: Target labels
clf: An untrained DecisionTreeClassifier instance

Returns:
A dictionary containing the best combination of hyperparameters found during
the grid search.
"""
from sklearn import model_selection


def prepruning(X, y, clf):
    """
    A function to perform a Grid Search for the best pre-pruning
    hyperparameters for a decision tree classifier.
    """

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 3, 4],
        'min_samples_leaf': [2, 3, 4],
        'min_samples_split': [2, 3, 4]
    }

    grid_search = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=param_grid
    )

    grid_search.fit(X, y)

    return grid_search.best_params_
