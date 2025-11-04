# Core libraries
import math

# Data manipulation
import pandas as pd        # DataFrames for structured data manipulation

# Visualization libraries
import matplotlib.pyplot as plt  # For plotting graphs and visualizations
import seaborn as sns             # Advanced visualization (heatmaps, pairplots, etc.)

# Set visualization style
sns.set(style="whitegrid")

# Data Visualization

# Plot Missing Values HEATMAP

def missing_values_heatmap(df, dataset_name="Dataset"):
    """
    Plots a heatmap of missing values (HEATMAP) for a given DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to visualize.
    dataset_name : str, optional
        Name of the dataset to be shown in the plot title (default = "Dataset").
    """
    plt.figure(figsize=(18, 6))
    sns.heatmap(
        df.isnull(),
        yticklabels=False,   # Hide row labels
        cbar=False,          # Remove color bar
        cmap="viridis"       # Colormap for missing values
    )

    plt.title(f"Heatmap of Missing Values - {dataset_name}", 
              fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Rows", fontsize=12)
    plt.show()

# Plot Missing Values BARCHART

def missing_values_barchart(df, dataset_name="Dataset", top_n=None):
    """
    Plots a bar chart of missing values (in %) for each feature in a DataFrame,
    with percentage values displayed above each bar.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to analyze.
    dataset_name : str, optional
        Name of the dataset for the plot title. Default is "Dataset".
    top_n : int, optional
        If specified, show only the top_n features with the most missing values.
    """
    # Calculate percentage of missing values per column
    missing_percent = df.isnull().mean() * 100
    
    # Filter out columns with 0% missing values
    missing_percent = missing_percent[missing_percent > 0]
    
    # Sort descending
    missing_percent = missing_percent.sort_values(ascending=False)
    
    # If top_n specified, take only top_n features
    if top_n is not None:
        missing_percent = missing_percent.head(top_n)
    
    if missing_percent.empty:
        print(f"No missing values in {dataset_name} dataset!")
        return
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(missing_percent.index, missing_percent.values, color='salmon', edgecolor='black')
    
    # Add percentage labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.title(f"Missing Values by Feature - {dataset_name}", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Missing Values (%)", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(missing_percent.values)*1.15)  # Add a little space above bars for labels
    plt.tight_layout()
    plt.show()


# Missing Values Summary

def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary of missing values for each column in a DataFrame, including the column type.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to analyze.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with one row per column containing:
        - Column: Name of the column
        - Dtype: Data type of the column
        - TotalValues: Total number of rows in the DataFrame
        - MissingValues: Number of missing (NaN) values in the column
        - NonMissingValues: Number of non-missing values in the column
        - MissingPercent: Percentage of missing values in the column
    """
    
    summary = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [df[col].dtype for col in df.columns],
        "TotalValues": len(df),
        "MissingValues": df.isnull().sum().values,
        "NonMissingValues": df.notnull().sum().values
    })
    
    summary["MissingPercent"] = (summary["MissingValues"] / summary["TotalValues"]) * 100
    
    return summary

