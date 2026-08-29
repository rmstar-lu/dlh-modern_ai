#!/usr/bin/env python3
"""
A function that initializes and returns an untrained boosting classifier based
on the specified algorithm name.

Initialize the selected model with the specified n_estimators and random_state.
For LightGBM, verbose is set to -1 to suppress training logs.
Raise a ValueError with the message “Unknown model name '{name}'” if the
provided model name is invalid.

Arguments:
name (str): Name of the boosting algorithm. Must be one of:

'adaboost' — returns an AdaBoostClassifier

'gradientboosting' — returns a GradientBoostingClassifier

'xgboost' — returns an XGBClassifier

'lightgbm' — returns an LGBMClassifier

n_estimators (int): Number of boosting iterations (trees).

random_state (int): Random seed for reproducibility.

Returns:
An untrained instance of the selected boosting classifier.
"""
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """
    Initialize and return an untrained boosting classifier based on the
    specified algorithm name
    """

    if name == "adaboost":
        return ensemble.AdaBoostClassifier(
                n_estimators=n_estimators,
                random_state=random_state
        )
    if name == "gradientboosting":
        return ensemble.GradientBoostingClassifier(
                n_estimators=n_estimators,
                random_state=random_state
        )
    if name == "xgboost":
        return xgb.XGBClassifier(
                n_estimators=n_estimators,
                random_state=random_state
        )
    if name == "lightgbm":
        return lgb.LGBMClassifier(
                verbose=-1,
                n_estimators=n_estimators,
                random_state=random_state
        )
    raise ValueError(f"Unknown model name '{name}'")
