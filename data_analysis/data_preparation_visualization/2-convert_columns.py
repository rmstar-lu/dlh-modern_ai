#!/usr/bin/env python3
"""
2-convert_columns.py: A function that performs type conversion for
specific columns:

df: pandas DataFrame containing the columns TotalCharges and SeniorCitizen
Converts the TotalCharges column to numeric.
Non-numeric entries should be converted to NaN
Maps the numeric values in the SeniorCitizen column (0 and 1) to categorical
strings "No" and "Yes" respectively
Returns: The modified DataFrame
"""

import pandas as pd


def convert_columns(df):
    """ Function to perform type conversion on 2 columns """

    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors="coerce")
    df.SeniorCitizen = df.SeniorCitizen.replace({0: "No", 1: "Yes"})
    return df
