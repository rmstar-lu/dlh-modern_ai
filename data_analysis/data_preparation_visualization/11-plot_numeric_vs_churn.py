#!/usr/bin/env python3
"""
Module 11-plot_numeric_vs_churn:
A function that compares continuous numeric feature distributions by churn:

df: pandas DataFrame with Churn column
col: Numeric column name
Uses a figure size of (12, 8)
Adds a title to the plot in the format: "<col> Distribution by Churn"
Plots overlapping histograms for Churn=Yes and Churn=No
Sets the x-axis label to "<col>"
Uses 30 bins to group the data along the x-axis
Adds a legend with a title
Displays the plot
Returns: None
"""
import pandas as pd
import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """
    Compare continuous numeric feature distributions by churn
    """
    plt.figure(figsize=(12, 8))

    plt.title(f"{col} Distribution by Churn")
    n = df[col].loc[df['Churn'] == 'No']
    y = df[col].loc[df['Churn'] == 'Yes']
    plt.hist((n, y), bins=30, label=['No', 'Yes'])
    plt.xlabel(col)
    plt.legend(title="Churn", loc="upper center")
    plt.show()
