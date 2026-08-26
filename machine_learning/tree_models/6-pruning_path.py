#!/usr/bin/env python3
"""
A function that retrieves the cost-complexity pruning path for a given
decision tree classifier.

Arguments:
clf: A DecisionTreeClassifier instance
X: Input features
y: Target labels

Returns:
ccp_alphas: A NumPy array containing the effective alpha values used for
pruning
impurities: A NumPy array containing the total impurity of leaves at each
corresponding alpha
"""


def get_pruning_path(clf, X, y):
    """
    Retrieve the cost-complexity pruning path for a given decision tree.
    """

    ccp_path = clf.cost_complexity_pruning_path(X, y)

    return (ccp_path.ccp_alphas, ccp_path.impurities)
