#!/usr/bin/env python3
""" 14-create_features module:
A function that engineers new features from the dataset:

df: pandas DataFrame

Creates:
NumServices: Number of services the customer is subscribed to (counting only
those with 'Yes' in selected service-related columns)
Do not include the PhoneService column, as it was dropped based on the
decision made in Task 12
For InternetService, count 'DSL' and 'Fiber optic' as 'Yes' (i.e., subscribed
to the service), and 'No' as not subscribed
TenureGroup: A categorical column that bins the tenure into intervals:
0-12, 13-24, 25-48, 49-60, 60+ , where 0 is excluded and upper bounds
are inclusive.
Drops the original columns that were used to create the new ones

Returns the modified DataFrame
"""
import pandas as pd


def create_features(df):
    """
    A function that engineers new features from the dataset
    """
    services = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['NumServices'] = (df[services] == 'Yes').sum(axis=1) + \
        1 * (df.InternetService == 'DSL') + \
        1 * (df.InternetService == 'Fiber optic')
    bins = [0, 12, 24, 48, 60, df.tenure.max()]
    names = ['0-12', '13-24', '25-48', '49-60', '60+']
    df['TenureGroup'] = pd.cut(df.tenure, bins, labels=names)
    df.drop(columns=services + ['tenure'], inplace=True)
    return df
