#!/usr/bin/env python3
"""
Module 8-plot_continuous_distributions.py:
A function that visualizes distributions of continuous numerical features:

df: pandas DataFrame
columns_to_plot: Optional list of continuous numeric columns to plot.
If None, it selects all numeric columns

For each selected column, generate:
Left subplot: Histogram with KDE using the following settings:
bins = 30
density = True
alpha = 0.7
edgecolor = 'black'
KDE line color should be red
Title format: "<column_name> Histogram + KDE"

Right subplot: Box Plot
Title format: "<column_name> Boxplot"
Displays the plot

Returns: None
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_continuous_distributions(df, columns_to_plot=None):
    """
    Visualize distributions of continuous numerical features
    """
    if columns_to_plot is None:
        columns_to_plot = df.select_dtypes(include='number').columns.values

    n_cols = max(1, len(columns_to_plot))
    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 3 * n_cols))
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, column in enumerate(columns_to_plot):
        ax = axes[2 * i]  # left subplot
        ax.set_title(f"{column} Histogram + KDE")
        kde = stats.gaussian_kde(df[column])
        x_vals = np.linspace(df[column].min(), df[column].max(), 200)
        ax.hist(df[column], bins=30, density=True,
                alpha=0.7, edgecolor='black')
        ax.plot(x_vals, kde(x_vals), color='red', linestyle='dashed')

        ax = axes[2 * i + 1]  # right subplot
        ax.set_title(f"{column} Boxplot")
        ax.boxplot(df[column], vert=False)

    for ax in axes[2 * len(columns_to_plot):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()
