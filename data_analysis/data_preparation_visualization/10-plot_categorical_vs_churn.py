#!/usr/bin/env python3
"""
Module 10-plot_categorical_vs_churn:
A function that visualizes churn rates per category:

df: pandas DataFrame with Churn column
col: Categorical column name
Uses a figure size of (12, 8)
Adds a title to the plot in the format: "Churn Rate by <col>"
Plots churn rate (Yes proportion) per category as a bar plot
Sets y-axis label to "Churn Rate"
Rotates the x-axis labels by 45°
Displays the plot
Returns: None
"""
import pandas as pd
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """
    Visualize churn rate per category
    """
    plt.figure(figsize=(12, 8))

    plt.title(f"Churn Rate by {col}")
    vc = df[col].value_counts().sort_index()
    vcy = df[col].loc[df['Churn'] == 'Yes'] \
        .value_counts().reindex(vc.index, fill_value=0)
    plt.bar(vcy.index, vcy / vc)
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)
    plt.show()
