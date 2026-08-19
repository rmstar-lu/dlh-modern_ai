#!/usr/bin/env python3
"""
Module 3-clean_total_charges.py:
A function that handles missing values in TotalCharges:

df: pandas DataFrame with missing values in TotalCharges
method: Strategy to handle missing values:
'drop': Remove rows with missing TotalCharges
'median': Fill with column median
'impute': Replace with MonthlyCharges * tenure
Returns the modified DataFrame
"""


def clean_total_charges(df, method='drop'):
    """ A function to handle missing values in TotalCharges """

    if method == 'drop':
        return df[df.TotalCharges.notna()]
    if method == 'median':
        fill = df.TotalCharges.median()
    elif method == 'impute':
        fill = df.MonthlyCharges * df.tenure
    else:
        raise ValueError("unsupported method")
    df = df.copy()
    df.TotalCharges = df.TotalCharges.fillna(fill)
    return df
