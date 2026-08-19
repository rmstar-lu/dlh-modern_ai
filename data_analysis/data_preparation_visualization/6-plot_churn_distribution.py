#!/usr/bin/env python3
"""
Module 6-plot_churn_distribution:
A function that visualizes churn class distribution:

df: pandas DataFrame with a Churn column
Generates a bar plot of Churn value counts
Uses colors: skyblue for ('No'), salmon for ('Yes')
Displays the plot
Returns: None
"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    Visualize churn class distribution.
    """
    plt.figure(figsize=(12, 8))

    plt.title("Churn Distribution")
    plt.ylabel("Count")
    vc = df.Churn.value_counts()
    colors = [{'No': 'skyblue', 'Yes': 'salmon'}[yn] for yn in vc.index]
    plt.bar(vc.index, vc, color=colors)

    plt.show()
