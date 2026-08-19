#!/usr/bin/env python3
"""
A function that visualizes missing values in a DataFrame:

df: pandas DataFrame to analyze

Generates a scatter plot where:

The x-axis represents row indices (DataFrame records)
The y-axis represents column names.
Y-tick labels are explicitly mapped to the DataFrame column names.
Each missing value is displayed as a vertical bar (|),
using the default plotting color.
Displays the plot using Matplotlib

Returns: None
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    A function that visualizes missing values in a DataFrame.
    """
    plt.figure(figsize=(12, 8))

    plt.title("Missingness plot")

    x, y = np.nonzero(df.isna().values)

    plt.scatter(x, y, marker="|")
    plt.yticks(np.arange(len(df.columns)), df.columns.values)

    plt.tight_layout()
    plt.show()
