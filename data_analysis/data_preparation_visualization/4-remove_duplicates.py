#!/usr/bin/env python3
"""
Module 4-remove_duplicates.py:
A function that removes duplicate rows:

df: pandas DataFrame to process
Drops all duplicate rows
Returns the deduplicated DataFrame
"""


def remove_duplicates(df):
    """ A function to remove duplicate rows """

    return df.drop_duplicates()
