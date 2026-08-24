#!/usr/bin/env python3
"""
Module 15-encode_features:
A function that encodes features for modeling using Scikit-learn:

df: pandas DataFrame

The function should encode:
Churn: LabelEncoder (No→0, Yes→1)
Partner, Dependents, PaperlessBilling, SeniorCitizen:
OrdinalEncoder (No→0, Yes→1)
Contract, PaymentMethod: One-hot encoding with drop first set to True
TenureGroup: Alphabetical order OrdinalEncoder

Returns:
The encoded DataFrame
The Fitted LabelEncoder for Churn
The Fitted OrdinalEncoder for binary columns
The Fitted OrdinalEncoder for TenureGroup
"""
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """ Encode features for modeling using Scikit-learn """
    churn_le = preprocessing.LabelEncoder()
    churn_le.fit(["No", "Yes"])
    df['Churn'] = churn_le.transform(df['Churn'])

    binary_oe = preprocessing.OrdinalEncoder(
            categories=[["No", "Yes"]])
    binary_cols = [
            'Partner', 'Dependents', 'PaperlessBilling', 'SeniorCitizen']
    for col in binary_cols:
        df[[col]] = binary_oe.fit_transform(df[[col]]).astype(int)

    onehot_cols = ['Contract', 'PaymentMethod']
    ohe = preprocessing.OneHotEncoder(drop='first', dtype=int)
    for col in onehot_cols:
        df[ohe.get_feature_names_out()] = \
            ohe.fit_transform(df[[col]]).toarray()
    df.drop(columns=onehot_cols, inplace=True)

    groups = list(df['TenureGroup'].unique())
    groups.sort()
    tenure_oe = preprocessing.OrdinalEncoder(
            categories=[groups], dtype=int)
    df[['TenureGroup']] = tenure_oe.fit_transform(df[['TenureGroup']])
    return df, churn_le, binary_oe, tenure_oe
