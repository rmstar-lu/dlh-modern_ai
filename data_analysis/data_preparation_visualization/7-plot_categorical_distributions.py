#!/usr/bin/env python3
"""
Module 7-plot_categorical_distributions:
A function that visualizes categorical feature distributions:

df: pandas DataFrame
columns_to_plot: Optional list of categorical columns (Default: all columns
with dtype object, excluding the target variable Churn.)
Generates bar plots for each categorical feature in a grid layout
Rotates x-axis labels by 45°
Displays the plot
Returns: None
"""
import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """
    Visualize categorical feature distributions.
    """
    if columns_to_plot is None:
        columns_to_plot = [c for c, t in zip(df.columns, df.dtypes)
                           if c != 'Churn' and t == 'object']
    if columns_to_plot:
        n_cols = min(3, len(columns_to_plot))
        n_rows = 1 + (len(columns_to_plot) - 1) // n_cols
    else:
        n_cols, n_rows = 1, 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns_to_plot):
        ax = axes[i]
        ax.set_title(column)
        vc = df[column].value_counts()
        ax.bar(vc.index, vc)
        ax.tick_params(axis='x', labelrotation=45)

    for ax in axes[len(columns_to_plot):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
