# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

# Plotting
def plot_histogram(df: pd.DataFrame, bins: int, column: str, log_scale: bool = False) -> None:
    """
    Plot histograms for a specific column, comparing normal and anomaly occurrences.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    bins : int
        Number of bins for the histogram.
    column : str
        Column name for which the histogram is plotted.
    log_scale : bool, optional
        Whether to use a log scale for the y-axis, by default False.
    """
    bins = 100

    anomalies = df[df['Class'] == 1]
    normal = df[df['Class'] == 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle(f'Counts of {column} by Class')

    ax1.hist(anomalies[column], bins=bins, color='red')
    ax1.set_title('Anomalies')

    ax2.hist(normal[column], bins=bins, color='blue')
    ax2.set_title('Normal')

    plt.xlabel(f'{column}')
    plt.ylabel('Count')
    if log_scale:
        plt.yscale('log')
    plt.xlim((np.min(df[column]), np.max(df[column])))
    plt.show()

    return None

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, sharey: bool = False) -> None:
    """
    Plot scatter plots for two columns, comparing normal and anomaly occurrences.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    x_col : str
        Column name for the x-axis.
    y_col : str
        Column name for the y-axis.
    sharey : bool, optional
        Whether to share the y-axis, by default False.
    """
    anomalies = df[df['Class'] == 1]
    normal = df[df['Class'] == 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=sharey)
    fig.suptitle(f'Scatter Plot of {x_col} and {y_col} by Class')

    ax1.scatter(anomalies[x_col], anomalies[y_col], color='red')
    ax1.set_title('Anomalies')

    ax2.scatter(normal[x_col], normal[y_col], color='blue')
    ax2.set_title('Normal')

    plt.xlabel(f'{x_col}')
    plt.ylabel(f'{y_col}')

    return None