#!/usr/bin/env python3
"""
Module 12-chi_square_tests:
A function that performs chi-square tests for categorical features:

df: pandas DataFrame with Churn and categorical columns

Computes the Chi-square p-value to test the independence between each
categorical feature and the target variable Churn, excluding Churn itself
from the features tested.

Returns a dictionary: {feature_name: p_value}

If the p_valuei for some feature is >= 0.05 that feature has no significant
relationship to the Churn variable.
"""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """
    Perform chi-square tests for categorical features
    """
    features = [c for c, t in zip(df.columns, df.dtypes)
                if c != 'Churn' and t == 'object']
    result = {}
    for col in features:
        obs = pd.crosstab(df[col], df['Churn'])
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(obs)
        result[col] = p_val
    return result
