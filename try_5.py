import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from sklearn.feature_extraction.text import CountVectorizer
import time
import pickle
import os

from Data_Preprocessing import visualize_data

# Constants
MODEL_FILE = r'D:\Internship\best_lora_model.pkl'

# Modulation mappings
MODULATION_FULL_NAMES = {
    'CSS': 'Chirp Spread Spectrum',
    'FSK': 'Frequency-Shift Keying',
    'LoRa': 'LoRa Proprietary Modulation'
}

# Synthetic Data Generator
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'node': np.random.choice(['Node1', 'Node2', 'Node3', 'Node4'], num_samples),
        'snr': np.random.uniform(5, 20, num_samples),
        'ber': np.random.uniform(0, 0.1, num_samples),
        'throughput': np.random.uniform(50, 150, num_samples),
        'modulation': np.random.choice(['CSS', 'FSK', 'LoRa'], num_samples),
        'signal_strength': np.random.uniform(0, 1, num_samples),
        'data_rate': np.random.choice(['DR0', 'DR1', 'DR2', 'DR3'], num_samples),
        'spreading_factor': np.random.choice([7, 8, 9, 10, 11, 12], num_samples),
        'bandwidth': np.random.choice([125, 250, 500], num_samples),
        'latency': np.random.uniform(10, 100, num_samples),
        'packet_loss': np.random.uniform(0, 0.1, num_samples),
        'distance_to_gateway': np.random.uniform(1, 20, num_samples)
    }
    return pd.DataFrame(data)

# Data Preprocessing
def preprocess_data(data, missing_strategy="mean", scaler_type="standard"):
    numerical_features = data.select_dtypes(include=[np.number])
    if missing_strategy == "mean":
        data[numerical_features.columns] = numerical_features.fillna(numerical_features.mean())
    elif missing_strategy == "median":
        data[numerical_features.columns] = numerical_features.fillna(numerical_features.median())
    elif isinstance(missing_strategy, (int, float)):
        data[numerical_features.columns] = numerical_features.fillna(missing_strategy)
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported scaler type")
    data[numerical_features.columns] = scaler.fit_transform(numerical_features)
    categorical_features = data.select_dtypes(include=[object])
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_categorical = encoder.fit_transform(categorical_features)
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features.columns))
    data = pd.concat([data[numerical_features.columns], encoded_categorical_df], axis=1)
    return data

# Exploratory Data Analysis
def eda(data):
    st.subheader("Exploratory Data Analysis")
    corr = data.corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig_corr)
    fig_dist = make_subplots(rows=1, cols=3, subplot_titles=("SNR Distribution", "BER Distribution", "Throughput Distribution"))
    fig_dist.add_trace(go.Histogram(x=data['snr'], nbinsx=30, name="SNR"), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=data['ber'], nbinsx=30, name="BER"), row=1, col=2)
    fig_dist.add_trace(go.Histogram(x=data['throughput'], nbinsx=30, name="Throughput"), row=1, col=3)
    fig_dist.update_layout(title_text="Feature Distributions", showlegend=False)
    st.plotly_chart(fig_dist)

# Main Function
def main():
    st.title("Advanced LoRa Network Analysis")
    original_nodes_df = generate_synthetic_data()
    st.write("Synthetic data generated.")
    visualize_data(original_nodes_df, x="snr", y="throughput", color="modulation", title="SNR vs Throughput")
    nodes_df = preprocess_data(original_nodes_df)
    eda(nodes_df)

if __name__ == "__main__":
    main()
