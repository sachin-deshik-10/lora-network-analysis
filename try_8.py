import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import scipy.signal as signal
import requests
import paho.mqtt.client as mqtt
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import pickle

# Constants
MODEL_FILE = r'D:\Internship\best_lora_model.pkl'
MODULATION_FULL_NAMES = {
    'CSS': 'Chirp Spread Spectrum',
    'FSK': 'Frequency-Shift Keying',
    'LoRa': 'LoRa Proprietary Modulation'
}

def generate_synthetic_data(num_samples=1000):
    """Create synthetic datasets for analyzing LoRa networks."""
    np.random.seed(42)
    data = {
        'node': np.random.choice(['Node1', 'Node2', 'Node3', 'Node4'], num_samples),
        'snr': np.random.uniform(5, 20, num_samples),
        'ber': np.random.uniform(0, 0.1, num_samples),
        'throughput': np.random.uniform(50, 150, num_samples),
        'modulation': np.random.choice(['CSS', 'FSK', 'LoRa'], num_samples),
        'signal_strength': np.random.uniform(0.5, 1, num_samples),
        'data_rate': np.random.choice(['DR0', 'DR1', 'DR2', 'DR3'], num_samples),
        'spreading_factor': np.random.choice([7, 8, 9, 10, 11, 12], num_samples),
        'bandwidth': np.random.choice([125, 250, 500], num_samples),
        'latency': np.random.uniform(10, 100, num_samples),
        'packet_loss': np.random.uniform(0, 0.1, num_samples),
        'distance_to_gateway': np.random.uniform(1, 20, num_samples)
    }
    return pd.DataFrame(data)

def preprocess_data(data, missing_strategy="mean", scaler_type="standard"):
    """Clean, transform, and scale the dataset."""
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

    # Encode categorical features with OneHotEncoder
    categorical_features = data.select_dtypes(include=[object])
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_categorical = encoder.fit_transform(categorical_features)
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features.columns))

    # Combine numerical and encoded features
    data = pd.concat([data[numerical_features.columns], encoded_categorical_df], axis=1)
    return data

def eda(data):
    """Perform exploratory data analysis, including visualizations."""
    st.subheader("Exploratory Data Analysis")
    corr = data.corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    fig_dist = make_subplots(rows=1, cols=3, subplot_titles=("SNR Distribution", "BER Distribution", "Throughput Distribution"))
    fig_dist.add_trace(go.Histogram(x=data['snr'], nbinsx=30, name="SNR"), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=data['ber'], nbinsx=30, name="BER"), row=1, col=2)
    fig_dist.add_trace(go.Histogram(x=data['throughput'], nbinsx=30, name="Throughput"), row=1, col=3)
    fig_dist.update_layout(title_text="Feature Distributions", showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

def train_and_evaluate_models(data, target):
    """Implement supervised learning algorithms."""
    X = data.drop(target, axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression()
    }

    param_grid = {
        "Random Forest": {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]},
        "Gradient Boosting": {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]},
        "Linear Regression": {}
    }

    performance = {}
    best_model = None
    best_model_name = None

    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grid[name], cv=3, scoring='r2', error_score='raise')
        grid_search.fit(X_train, y_train)
        current_best_model = grid_search.best_estimator_
        predictions = current_best_model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        performance[name] = {"R2 Score": r2, "MSE": mse}
        st.write(f"{name} - Best Params: {grid_search.best_params_} - R2 Score: {r2:.2f}, MSE: {mse:.2f}")

        if hasattr(current_best_model, 'feature_importances_'):
            feature_importance_plot(X.columns, current_best_model.feature_importances_, name)

        if best_model is None or r2 > performance[best_model_name]["R2 Score"]:
            best_model_name = name
            best_model = current_best_model

    st.write(f"Best Model: {best_model_name} with R2 Score: {performance[best_model_name]['R2 Score']:.2f}")

    # Save the best model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(best_model, f)
    st.write("Best model saved for future use.")

    return best_model

def feature_importance_plot(features, importances, model_name):
    """Visualize feature importance for supported models."""
    fig = px.bar(x=features, y=importances, labels={'x': 'Feature', 'y': 'Importance'}, title=f"{model_name} Feature Importance")
    st.plotly_chart(fig)

def visualize_data(data, x="snr", y="throughput", color="modulation", title="Scatter Plot of SNR vs Throughput", theme="plotly"):
    """Create scatter plots for data visualization."""
    fig = px.scatter(data, x=x, y=y, color=color, title=title, template=theme)
    st.plotly_chart(fig)

def analyze_logs(logs, ngram_range=(1, 1)):
    """Conduct log analysis using NLP techniques."""
    logs = [re.sub(r'\W+', ' ', log) for log in logs]
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(logs)
    return X

def compare_signals(data, original_data):
    """Evaluate SNR, BER, and throughput to identify the best signal."""
    data_normalized = data[['snr', 'ber', 'throughput']].copy()
    data_normalized['snr'] = MinMaxScaler().fit_transform(data[['snr']])
    data_normalized['ber'] = MinMaxScaler().fit_transform(data[['ber']])
    data_normalized['throughput'] = MinMaxScaler().fit_transform(data[['throughput']])
    data_normalized['score'] = data_normalized['snr'] + data_normalized['throughput'] - data_normalized['ber']
    best_signal_index = data_normalized['score'].idxmax()
    best_signal = original_data.iloc[best_signal_index]
    return best_signal

def analyze_modulation(data):
    """Analyze and summarize data based on modulation schemes."""
    modulation_summary = data.groupby('modulation').agg({
        'snr': 'mean',
        'ber': 'mean',
        'throughput': 'mean',
        'data_rate': lambda x: x.mode()[0],
        'spreading_factor': 'mean',
        'bandwidth': 'mean',
        'latency': 'mean',
        'packet_loss': 'mean',
        'distance_to_gateway': 'mean'
    }).reset_index()

    st.subheader("Modulation Analysis")
    modulation_summary['modulation_full'] = modulation_summary['modulation'].map(MODULATION_FULL_NAMES)
    st.write(modulation_summary)

    best_modulation = modulation_summary.loc[modulation_summary['throughput'].idxmax()]
    st.write(f"The best modulation based on throughput: {best_modulation['modulation_full']} with average SNR: {best_modulation['snr']}, BER: {best_modulation['ber']}, throughput: {best_modulation['throughput']}, latency: {best_modulation['latency']} ms, and packet loss: {best_modulation['packet_loss']}.")

    return best_modulation['modulation']

def display_waveform(modulation):
    """Visualize the waveform of a modulation type."""
    t = np.linspace(0, 1, 1000)
    if modulation == "BPSK":
        waveform = np.sign(np.sin(2 * np.pi * 5 * t))
    elif modulation == "QPSK":
        waveform = np.sin(2 * np.pi * 5 * t) + np.cos(2 * np.pi * 5 * t)
    elif modulation == "8-PSK":
        waveform = np.sin(2 * np.pi * 5 * t) + np.cos(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
    elif modulation == "CSS":
        waveform = signal.chirp(t, f0=6, f1=1, t1=1, method='linear')
    else:
        waveform = np.sin(2 * np.pi * 5 * t)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, waveform)
    ax.set_title(f'Waveform for {modulation}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    st.pyplot(fig)

def estimate_performance(snr, distance, temperature, humidity, payload_size, modulation):
    """Estimate performance metrics such as throughput and latency."""
    modulation_gain = {"BPSK": 1, "QPSK": 2, "8-PSK": 3, "CSS": 1.5}
    gain_factor = modulation_gain.get(modulation, 1)
    throughput = max(0, (snr - 7) * 0.1 * gain_factor + np.random.uniform(0.1, 0.5) - payload_size * 0.01)
    latency = (distance / 100) * 50 + np.random.uniform(5, 20) + payload_size * 0.2 / gain_factor
    return throughput, latency

def main():
    st.title("Comprehensive Wireless Communication Simulation and Analysis")

    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["EDA & Model Training", "Real-Time Simulations", "Networking Insights"])

    if page == "EDA & Model Training":
        # Generate synthetic data
        original_nodes_df = generate_synthetic_data()
        st.write("Synthetic data generated for analysis.")

        # Preprocess data for model training
        nodes_df = preprocess_data(original_nodes_df)

        # Perform EDA
        eda(nodes_df)

        st.subheader("Train Machine Learning Model")
        if st.button("Train Model"):
            best_model = train_and_evaluate_models(nodes_df, 'signal_strength')

    elif page == "Real-Time Simulations":
        # Real-Time Performance Simulation
        st.subheader("Performance Simulation")
        snr = st.slider("SNR (dB)", min_value=0, max_value=20, value=10, key="snr_slider")
        distance = st.slider("Distance (m)", min_value=1, max_value=1000, value=500, key="distance_slider")
        temperature = st.slider("Temperature (°C)", min_value=-10, max_value=50, value=25, key="temperature_slider")
        humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50, key="humidity_slider")
        payload_size = st.slider("Payload Size (bytes)", min_value=10, max_value=255, value=50, key="payload_size_slider")
        modulation = st.selectbox("Modulation Type", ["BPSK", "QPSK", "8-PSK", "CSS"], key="modulation_selectbox")

        if st.button("Run Simulation", key="run_simulation_button"):
            throughput, latency = estimate_performance(snr, distance, temperature, humidity, payload_size, modulation)
            st.metric("Throughput (kbps)", f"{throughput:.2f}")
            st.metric("Latency (ms)", f"{latency:.2f}")

            # Visualization
            display_waveform(modulation)

    elif page == "Networking Insights":
        st.subheader("Protocol Simulations")
        protocol = st.selectbox("Select Protocol", ["TCP", "UDP", "HTTP", "FTP"], key="protocol_selectbox")
        packet_loss = st.slider("Packet Loss (%)", min_value=0, max_value=100, value=0, key="packet_loss_slider")
        transmission_delay = st.slider("Transmission Delay (ms)", min_value=0, max_value=100, value=10, key="transmission_delay_slider")
        bandwidth = st.slider("Bandwidth (Mbps)", min_value=1, max_value=100, value=10, key="bandwidth_slider")

        if st.button("Run Protocol Simulation", key="run_protocol_simulation_button"):
            st.write(f"Simulating {protocol} with {packet_loss}% packet loss, {transmission_delay}ms delay, and {bandwidth}Mbps bandwidth.")
            # Placeholder for actual protocol simulation logic
            st.write("Simulation complete")

            # Visualization: Bandwidth usage bar chart
            fig, ax = plt.subplots()
            used_bandwidth = bandwidth * (1 - packet_loss / 100)
            available_bandwidth = bandwidth - used_bandwidth
            ax.bar(["Used Bandwidth", "Available Bandwidth"], [used_bandwidth, available_bandwidth], color=['blue', 'red'])
            ax.set_title("Bandwidth Usage")
            ax.set_ylabel("Mbps")
            ax.set_ylim(0, bandwidth)
            ax.grid(axis='y')
            st.pyplot(fig)

if __name__ == "__main__":
    main()