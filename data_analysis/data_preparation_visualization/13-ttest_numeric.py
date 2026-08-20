#!/usr/bin/env python3
"""
Module 13-ttest_numeric:
A function that performs Welch's t-tests for continuous numeric features

df: pandas DataFrame with Churn column

Computes t-test p-value comparing Churn=Yes vs Churn=No for each numeric feature

The Hypothesis being tested is:

H_0 (null): The means of the variable are equal in Churn=Yes and Churn=No groups
H_1 (alternative): The means differ significantly
Returns a dictionary: {feature_name: p_value}

Welch's t-test does not assume equal variance between the two groups.
"""
from scipy import stats


def ttest_numeric(df):
    """
    Perform Welch's t-tests for continuous numeric features
    """
    features = df.select_dtypes(include='number').columns.values
    result = {}
    for col in features:
        group_a = df[col].loc[df.Churn == 'No']
        group_b = df[col].loc[df.Churn == 'Yes']
        t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
        result[col] = p_val
    return result
