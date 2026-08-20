#!/usr/bin/env python3
"""
Module 9-plot_correlation_heatmap.py:
A function that visualizes correlations between continuous numeric
features using seaborn:

df: pandas DataFrame
Computes pairwise correlations
Generates an annotated heatmap with coolwarm colormap
Set vmin = -1 and vmax = 1 so that the heatmap color mapping reflects
the full correlation range
Displays the plot
Returns: None
"""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    Visualize correlations between continuous numeric features
    """
    plt.figure(figsize=(6, 5))

    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=True)
    plt.title("Correlation Matrix")
    plt.show()
