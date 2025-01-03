#%%
# setup_utils.py
import os
import logging
import json
import math
import time
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# Optional: If you use these in multiple notebooks
import umap.umap_ as umap
from sklearn.cluster import KMeans, DBSCAN

# Load environment variables
load_dotenv(verbose=True, encoding='utf-8')

def setup_db_connection():
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    db_conn_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    db_conn_query = """
    select implementation_info_name key, character_value as value
    from mimiciv.information_schema.sql_implementation_info
    where implementation_info_name like 'DBMS %'
    """

    try:
        engine = create_engine(db_conn_string)
        with engine.connect() as conn:
            conn.execute(text(db_conn_query))
        return engine
    except Exception as e:
        raise

def plr():
    local_time = datetime.now().astimezone()
    formatted_time = local_time.strftime('%y/%m/%d %H:%M:%S %Z')
    print(f"@{formatted_time}")

def save_intermediate_data(df, filename):
    """Save intermediate DataFrame to CSV or Pickle."""
    df.to_csv(filename, index=False)

def load_intermediate_data(filename):
    """Load intermediate DataFrame from CSV or Pickle."""
    return pd.read_csv(filename)

########################
# Generic Plot Functions
########################
def plot_histogram(data, column, bins=20, title=None, xlabel=None, ylabel="Frequency"):
    plt.figure(figsize=(8, 6))
    plt.hist(data[column].dropna(), bins=bins, color='blue', edgecolor='black')
    plt.title(title if title else f"Histogram of {column}")
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_scatter(data, x_col, y_col, title=None, xlabel=None, ylabel=None, alpha=0.7):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[x_col], data[y_col], alpha=alpha, color='blue')
    plt.title(title if title else f"Scatter Plot of {y_col} vs {x_col}")
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_line(data, x_col, y_col, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(8, 6))
    plt.plot(data[x_col], data[y_col], color='blue', marker='o', linestyle='-')
    plt.title(title if title else f"Line Plot of {y_col} over {x_col}")
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_correlation_matrix(corr_matrix, cmap="coolwarm", annot=True):
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, cbar=True, square=True, fmt=".2f",
                xticklabels=corr_matrix.columns, yticklabels=corr_matrix.index)
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.show()

def plot_forecast(final_results, subject_id, metric, forecast_horizon=None):
    """
    Plots historical and forecasted data for a given subject and metric.
    If forecast_horizon is specified, limits forecast data to that many days
    after the last historical date (if historical data exists).

    Parameters
    ----------
    final_results : pd.DataFrame
        A DataFrame containing historical and forecasted data.
        Expected columns: ['charttime', 'subject_id', 'metric', 'forecast',
                           'forecast_lower', 'forecast_upper', 'row_type'].
        'row_type' should be 'historical' or 'forecast'.

    subject_id : int or str
        The subject_id for which to plot the data.

    metric : str
        The metric to plot.

    forecast_horizon : int, optional
        Number of days of forecast data to plot beyond the last historical date.
        If None, plots all forecasted data available.
    """

    # Filter results for the given subject_id and metric
    df_sub = final_results[
        (final_results['subject_id'] == subject_id) &
        (final_results['metric'] == metric)
        ].copy()

    if df_sub.empty:
        print(f"No data found for subject_id={subject_id}, metric={metric}")
        return

    # Sort by charttime to ensure proper temporal ordering
    df_sub = df_sub.sort_values('charttime')

    # Separate historical and forecast data
    hist_df = df_sub[df_sub['row_type'] == 'historical']
    fc_df = df_sub[df_sub['row_type'] == 'forecast']

    if not hist_df.empty and forecast_horizon is not None:
        # If we have historical data and a forecast horizon
        last_hist_date = hist_df['charttime'].max()
        cutoff_date = last_hist_date + pd.Timedelta(days=forecast_horizon)
        fc_df = fc_df[fc_df['charttime'] <= cutoff_date]

    elif hist_df.empty and forecast_horizon is not None and not fc_df.empty:
        # If no historical data but a horizon is given
        first_fc_date = fc_df['charttime'].min()
        cutoff_date = first_fc_date + pd.Timedelta(days=forecast_horizon)
        fc_df = fc_df[fc_df['charttime'] <= cutoff_date]

    plt.figure(figsize=(12, 6))

    # Plot historical data
    if not hist_df.empty:
        plt.plot(hist_df['charttime'], hist_df[metric], label='Historical', color='blue', linewidth=2)
        # Draw a vertical line at the end of historical data (optional)
        plt.axvline(x=hist_df['charttime'].max(), color='gray', linestyle='--', linewidth=1, label='History/Forecast Boundary')

    # Plot forecasted values
    if not fc_df.empty:
        plt.plot(fc_df['charttime'], fc_df['forecast'], label='Forecast', color='red', linewidth=2)

        # Plot the forecast confidence interval as a shaded area, if available
        if 'forecast_lower' in fc_df.columns and 'forecast_upper' in fc_df.columns:
            plt.fill_between(fc_df['charttime'], fc_df['forecast_lower'], fc_df['forecast_upper'],
                             color='red', alpha=0.2, label='Confidence Interval')

    plt.xlabel('Time')
    plt.ylabel(metric.capitalize())
    horizon_str = f" (Horizon: {forecast_horizon} days)" if forecast_horizon is not None else ""
    plt.title(f"Subject {subject_id} - {metric.capitalize()} Forecast{horizon_str}")
    plt.legend()
    plt.tight_layout()
    plt.show()
