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

