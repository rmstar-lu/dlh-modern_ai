#!/usr/bin/env python3
"""
Module 17-split_data:
A function that splits data into train/test sets

df: pandas DataFrame
target: Name of target column
test_size: Proportion for test split
random_state: Random seed
Uses stratified sampling to preserve class distribution
Returns: (X_train, X_test, y_train, y_test)
"""
from sklearn import model_selection


def split_data(df, target='Churn', test_size=0.2, random_state=42):
    """ Split data into train/test sets """

    X, y = df.drop(columns=['Churn']), df['Churn']
    return model_selection.train_test_split(X, y,
                                            test_size=test_size,
                                            random_state=random_state,
                                            stratify=y)
