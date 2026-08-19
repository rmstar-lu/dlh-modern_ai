#!/usr/bin/env python3
"""
Module 5-drop_customerID:
A function that removes the customerID column
since unique identifiers lack predictive value

df: pandas DataFrame containing a customerID column
Drops the customerID column
Returns the modified DataFrame
"""


def drop_customerID(df):
    """ A function to drop the customerID column """

    return df.drop(['customerID'], axis=1)
