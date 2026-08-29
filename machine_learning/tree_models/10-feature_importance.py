#!/usr/bin/env python3
"""
A function that computes and returns the feature importances from a trained
random forest model.

Arguments:
rf: A trained Scikit-learn RandomForestClassifier instance.

Returns:
importances: A NumPy array of feature importance scores.
indices: A NumPy array of feature indices sorted from least to most important
(ascending order).
"""
import numpy as np


def feature_importance(rf):
    """
    Compute and return the feature importances from a trained
    random forest model
    """

    return (rf.feature_importances_, np.argsort(rf.feature_importances_))
