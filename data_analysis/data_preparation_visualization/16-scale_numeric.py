#!/usr/bin/env python3
"""
Module 16-scale_numeric:
A function that standardizes numeric columns

df: pandas DataFrame

Scales MonthlyCharges and TotalCharges using StandardScaler (mean=0, std=1)

Returns the modified DataFrame
"""
from sklearn import preprocessing


def scale_numeric(df):
    """ Standardize numeric columns """

    ss = preprocessing.StandardScaler()
    numeric_cols = ['MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        df[[col]] = ss.fit_transform(df[[col]])

    return df
