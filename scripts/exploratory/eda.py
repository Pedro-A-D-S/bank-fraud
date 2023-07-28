# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

# Setup
rcParams['figure.figsize'] = 14, 8

# Get Data
df = pd.read_csv('../../data/raw/archive/creditcard.csv')

# Get main informations about data
df.head()  # Display the first few rows of the DataFrame

df.info()  # Display information about the DataFrame, including data types and non-null counts

df.isnull().sum()  # Check for missing values in each column

df.describe()  # Generate summary statistics for numerical columns

df.shape  # Print the shape of the DataFrame (rows, columns)

# Divide into normal and anomaly occurrences
anomalies = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

print(f'Anomalies: {anomalies.shape}')
print(f'Normal: {normal.shape}')

# Look at the disparity in a graphical manner
class_counts = pd.value_counts(df['Class'], sort=True)
class_counts.plot(kind='bar', rot=0)
plt.title('Class Distribution')
plt.xticks(range(2), ['Normal', 'Anomaly'])
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig('../../data/images/class_distribuition.png')

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

# Plot the amount by Class for the entire dataframe
plt.scatter(df['Amount'], df['Class'])
plt.title('Transaction Amounts by Class')
plt.ylabel('Class')
plt.yticks(range(2), ['Normal', 'Anomaly'])
plt.xlabel('Transaction Amounts ($)')
plt.show()
plt.savefig('../../data/images/transaction_amounts_by_class.png')

# Plot histograms and scatter plots for various columns
bins = 100

plot_histogram(df=df, bins=bins, column='Amount', log_scale=True)

plt.hist(anomalies['Amount'], bins=bins, color='red')
plt.show()

plot_scatter(df, 'Time', 'Amount')

plot_histogram(df, bins, 'V1')

plot_scatter(df, 'Amount', 'V1', sharey=True)

plot_scatter(df, 'Time', 'V1', sharey=True)

for f in range(1, 29):
    print(f'V{f} Counts')
    plot_histogram(df, bins, f'V{f}')

for f in range(1, 29):
    print(f'V{f} vs Time')
    plot_scatter(df, 'Time', f'V{f}', sharey=True)

for f in range(1, 29):
    print(f'Amount vs V{f}')
    plot_scatter(df, 'Amount', f'V{f}', sharey=True)
