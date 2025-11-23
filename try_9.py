import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import scipy.signal as signal
import paho.mqtt.client as mqtt
import time
import json
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants
MODEL_FILE_PATH = r'D:\Internship\best_lora_model.pkl'
MODULATION_DESCRIPTIONS = {
    'CSS': 'Chirp Spread Spectrum',
    'FSK': 'Frequency-Shift Keying',
    'LoRa': 'LoRa Proprietary Modulation',
    'BPSK': 'Binary Phase Shift Keying',
    'QPSK': 'Quadrature Phase Shift Keying',
    '8-PSK': '8 Phase Shift Keying'
}
PROTOCOLS = ['LoRaWAN', 'MQTT', 'CoAP', 'HTTP']

def generate_synthetic_dataset(sample_size=1000):
    """Create synthetic datasets for analyzing LoRa networks."""
    np.random.seed(42)
    data = {
        'Node': np.random.choice(['Node1', 'Node2', 'Node3', 'Node4'], sample_size),
        'SNR': np.random.uniform(5, 20, sample_size),  # Typical SNR range for LoRa
        'BER': np.random.uniform(0, 0.1, sample_size),
        'Throughput': np.random.uniform(50, 150, sample_size),
        'Modulation': np.random.choice(list(MODULATION_DESCRIPTIONS.keys()), sample_size),
        'SignalStrength': np.random.uniform(0.5, 1, sample_size),
        'DataRate': np.random.choice(['DR0', 'DR1', 'DR2', 'DR3'], sample_size),
        'SpreadingFactor': np.random.choice([7, 8, 9, 10, 11, 12], sample_size),
        'Bandwidth': np.random.choice([125, 250, 500], sample_size),  # Common LoRa bandwidths
        'Latency': np.random.uniform(10, 100, sample_size),
        'PacketLoss': np.random.uniform(0, 0.1, sample_size),
        'DistanceToGateway': np.random.uniform(1, 20, sample_size)  # Typical LoRa range in km
    }
    return pd.DataFrame(data)

def preprocess_dataset(dataframe, missing_strategy="mean", scaler_type="standard"):
    """Clean, transform, and scale the dataset."""
    numerical_features = dataframe.select_dtypes(include=[np.number])

    if missing_strategy == "mean":
        dataframe[numerical_features.columns] = numerical_features.fillna(numerical_features.mean())
    elif missing_strategy == "median":
        dataframe[numerical_features.columns] = numerical_features.fillna(numerical_features.median())
    elif isinstance(missing_strategy, (int, float)):
        dataframe[numerical_features.columns] = numerical_features.fillna(missing_strategy)

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported scaler type")

    dataframe[numerical_features.columns] = scaler.fit_transform(numerical_features)

    # Encode categorical features with OneHotEncoder
    categorical_features = dataframe.select_dtypes(include=[object])
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_categorical = encoder.fit_transform(categorical_features)
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features.columns))

    # Combine numerical and encoded features
    dataframe = pd.concat([dataframe[numerical_features.columns], encoded_categorical_df], axis=1)
    return dataframe

def exploratory_data_analysis(dataframe):
    """Perform exploratory data analysis, including visualizations."""
    st.subheader("Exploratory Data Analysis")
    correlation_matrix = dataframe.corr()
    fig_corr = px.imshow(correlation_matrix, text_auto=True, title="Correlation Matrix", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    fig_dist = make_subplots(rows=1, cols=3, subplot_titles=("SNR Distribution", "BER Distribution", "Throughput Distribution"))
    fig_dist.add_trace(go.Histogram(x=dataframe['SNR'], nbinsx=30, name="SNR"), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=dataframe['BER'], nbinsx=30, name="BER"), row=1, col=2)
    fig_dist.add_trace(go.Histogram(x=dataframe['Throughput'], nbinsx=30, name="Throughput"), row=1, col=3)
    fig_dist.update_layout(title_text="Feature Distributions", showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

def train_and_evaluate_models(dataframe, target):
    """Implement supervised learning algorithms."""
    X = dataframe.drop(target, axis=1)
    y = dataframe[target]

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
            plot_feature_importance(X.columns, current_best_model.feature_importances_, name)

        if best_model is None or r2 > performance[best_model_name]["R2 Score"]:
            best_model_name = name
            best_model = current_best_model

    st.write(f"Best Model: {best_model_name} with R2 Score: {performance[best_model_name]['R2 Score']:.2f}")

    # Save the best model
    with open(MODEL_FILE_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    st.write("Best model saved for future use.")

    return best_model

def plot_feature_importance(features, importances, model_name):
    """Visualize feature importance for supported models."""
    fig = px.bar(x=features, y=importances, labels={'x': 'Feature', 'y': 'Importance'}, title=f"{model_name} Feature Importance")
    st.plotly_chart(fig)

def visualize_scatter(dataframe, x="SNR", y="Throughput", color="Modulation", title="Scatter Plot of SNR vs Throughput", theme="plotly"):
    """Create scatter plots for data visualization."""
    fig = px.scatter(dataframe, x=x, y=y, color=color, title=title, template=theme)
    st.plotly_chart(fig)

def analyze_logs(log_entries, ngram_range=(1, 1)):
    """Conduct log analysis using NLP techniques."""
    logs = [re.sub(r'\W+', ' ', log) for log in log_entries]
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(logs)
    return X

def compare_signal_quality(dataframe, original_data):
    """Evaluate SNR, BER, and throughput to identify the best signal."""
    normalized_data = dataframe[['SNR', 'BER', 'Throughput']].copy()
    normalized_data['SNR'] = MinMaxScaler().fit_transform(dataframe[['SNR']])
    normalized_data['BER'] = MinMaxScaler().fit_transform(dataframe[['BER']])
    normalized_data['Throughput'] = MinMaxScaler().fit_transform(dataframe[['Throughput']])
    normalized_data['Score'] = normalized_data['SNR'] + normalized_data['Throughput'] - normalized_data['BER']
    best_signal_index = normalized_data['Score'].idxmax()
    best_signal = original_data.iloc[best_signal_index]
    return best_signal

def analyze_modulation_schemes(dataframe):
    """Analyze and summarize data based on modulation schemes."""
    modulation_summary = dataframe.groupby('Modulation').agg({
        'SNR': 'mean',
        'BER': 'mean',
        'Throughput': 'mean',
        'DataRate': lambda x: x.mode()[0],
        'SpreadingFactor': 'mean',
        'Bandwidth': 'mean',
        'Latency': 'mean',
        'PacketLoss': 'mean',
        'DistanceToGateway': 'mean'
    }).reset_index()

    st.subheader("Modulation Analysis")
    modulation_summary['ModulationDescription'] = modulation_summary['Modulation'].map(MODULATION_DESCRIPTIONS)
    st.write(modulation_summary)

    best_modulation = modulation_summary.loc[modulation_summary['Throughput'].idxmax()]
    st.write(f"The best modulation based on throughput: {best_modulation['ModulationDescription']} with average SNR: {best_modulation['SNR']}, BER: {best_modulation['BER']}, throughput: {best_modulation['Throughput']}, latency: {best_modulation['Latency']} ms, and packet loss: {best_modulation['PacketLoss']}.")

    return best_modulation['Modulation']

def display_modulation_waveform(modulation_type):
    """Visualize the waveform of a modulation type."""
    t = np.linspace(0, 1, 1000)
    if modulation_type == "BPSK":
        waveform = np.cos(2 * np.pi * 5 * t)
    elif modulation_type == "QPSK":
        waveform = np.cos(2 * np.pi * 5 * t) + np.cos(2 * np.pi * 5 * t + np.pi/2)
    elif modulation_type == "8-PSK":
        waveform = np.cos(2 * np.pi * 5 * t) + np.cos(2 * np.pi * 5 * t + np.pi/4)
    elif modulation_type == "CSS":
        waveform = signal.chirp(t, f0=2, f1=10, t1=1, method='linear')
    else:
        waveform = np.cos(2 * np.pi * 5 * t)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, waveform)
    ax.set_title(f'Waveform for {modulation_type}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    st.pyplot(fig)

def estimate_network_performance(snr, distance, temperature, humidity, payload_size, modulation_type):
    """Estimate performance metrics such as throughput and latency."""
    modulation_gain = {"BPSK": 1, "QPSK": 2, "8-PSK": 3, "CSS": 1.5}
    gain_factor = modulation_gain.get(modulation_type, 1)
    throughput = max(0, (snr - 7) * 0.1 * gain_factor + np.random.uniform(0.1, 0.5) - payload_size * 0.01)
    latency = (distance / 100) * 50 + np.random.uniform(5, 20) + payload_size * 0.2 / gain_factor
    return throughput, latency

def setup_mqtt_client():
    """Setup MQTT client for real-time data fetching."""
    mqtt_client = mqtt.Client()

    def on_connect(client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        client.subscribe("lora/sensor/data")

    def on_message(client, userdata, msg):
        data = json.loads(msg.payload.decode())
        st.session_state['mqtt_data'] = data

    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect("broker.hivemq.com", 1883, 60)
    mqtt_client.loop_start()
    return mqtt_client

def visualize_signal_vs_performance(dataframe):
    """Visualize signal performance against SNR, BER, and Throughput across modulation schemes."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("SNR vs Modulation", "BER vs Modulation", "Throughput vs Modulation"))

    for modulation in dataframe['Modulation'].unique():
        mod_data = dataframe[dataframe['Modulation'] == modulation]
        fig.add_trace(go.Box(y=mod_data['SNR'], name=f"SNR-{modulation}"), row=1, col=1)
        fig.add_trace(go.Box(y=mod_data['BER'], name=f"BER-{modulation}"), row=1, col=2)
        fig.add_trace(go.Box(y=mod_data['Throughput'], name=f"Throughput-{modulation}"), row=1, col=3)

    fig.update_layout(title_text="Signal Performance by Modulation Scheme", showlegend=False)
    st.plotly_chart(fig)

# New function for signal modulation
def generate_signal(frequency, amplitude, duration, modulation_type="BPSK"):
    t = np.linspace(0, duration, int(frequency * duration), endpoint=False)
    if modulation_type == "BPSK":
        data = np.random.choice([-1, 1], size=t.shape)
        modulated_signal = amplitude * data * np.sin(2 * np.pi * frequency * t)
    elif modulation_type == "QPSK":
        data = np.random.choice([0, 1], size=(2, t.size))
        modulated_signal = amplitude * np.sin(2 * np.pi * frequency * t + np.pi * (data[0] + data[1]))
    else:
        raise ValueError("Unsupported modulation type!")
    return t, modulated_signal

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["EDA & Model Training", "Real-Time Simulations", "Networking Insights", "Signal vs Performance", "MQTT Example"])
    
    # Ensure original_data is initialized at the start
    original_data = generate_synthetic_dataset()

    if page == "EDA & Model Training":
        st.write("Synthetic data generated for analysis.")
        processed_data = preprocess_dataset(original_data)
        exploratory_data_analysis(processed_data)
        st.subheader("Train Machine Learning Model")
        if st.button("Train Model"):
            best_model = train_and_evaluate_models(processed_data, 'SignalStrength')

    elif page == "Real-Time Simulations":
        st.subheader("Performance Simulation")
        snr = st.slider("SNR (dB)", min_value=5, max_value=20, value=10, key="snr_slider")
        distance = st.slider("Distance (km)", min_value=1, max_value=20, value=5, key="distance_slider")
        temperature = st.slider("Temperature (°C)", min_value=-10, max_value=50, value=25, key="temperature_slider")
        humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50, key="humidity_slider")
        payload_size = st.slider("Payload Size (bytes)", min_value=10, max_value=255, value=50, key="payload_size_slider")
        modulation_type = st.selectbox("Modulation Type", list(MODULATION_DESCRIPTIONS.keys()), key="modulation_selectbox")

        if st.button("Run Simulation", key="run_simulation_button"):
            throughput, latency = estimate_network_performance(snr, distance, temperature, humidity, payload_size, modulation_type)
            st.metric("Throughput (kbps)", f"{throughput:.2f}")
            st.metric("Latency (ms)", f"{latency:.2f}")
            display_modulation_waveform(modulation_type)

        # Signal Modulation and Visualization
        st.subheader("Signal Modulation and Comparison")
        frequency = st.slider("Frequency (Hz)", 1, 100, 10)
        amplitude = st.slider("Amplitude", 1, 10, 5)
        duration = st.slider("Duration (s)", 1, 10, 2)
        mod_signal_type = st.selectbox("Modulation Type for Signal", ["BPSK", "QPSK"])

        t, signal = generate_signal(frequency, amplitude, duration, mod_signal_type)
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal, label=f"{mod_signal_type} Signal")
        plt.title(f"{mod_signal_type} Modulation")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        st.pyplot(plt)

    elif page == "Networking Insights":
        st.subheader("Protocol Simulations")
        protocol = st.selectbox("Select Protocol", PROTOCOLS, key="protocol_selectbox")
        packet_loss = st.slider("Packet Loss (%)", min_value=0, max_value=100, value=0, key="packet_loss_slider")
        transmission_delay = st.slider("Transmission Delay (ms)", min_value=0, max_value=100, value=10, key="transmission_delay_slider")
        bandwidth = st.slider("Bandwidth (Mbps)", min_value=1, max_value=100, value=10, key="bandwidth_slider")

        mqtt_client = setup_mqtt_client()

        if st.button("Run Protocol Simulation", key="run_protocol_simulation_button"):
            st.write(f"Simulating {protocol} with {packet_loss}% packet loss, {transmission_delay}ms delay, and {bandwidth}Mbps bandwidth.")
            st.write("Simulation complete")

            if 'mqtt_data' in st.session_state:
                st.write("MQTT Data:", st.session_state['mqtt_data'])

            fig, ax = plt.subplots()
            used_bandwidth = bandwidth * (1 - packet_loss / 100)
            available_bandwidth = bandwidth - used_bandwidth
            ax.bar(["Used Bandwidth", "Available Bandwidth"], [used_bandwidth, available_bandwidth], color=['blue', 'red'])
            ax.set_title("Bandwidth Usage")
            ax.set_ylabel("Mbps")
            ax.set_ylim(0, bandwidth)
            ax.grid(axis='y')
            st.pyplot(fig)

    elif page == "Signal vs Performance":
        st.subheader("Signal Performance by Modulation Scheme")
        visualize_signal_vs_performance(original_data)

    elif page == "MQTT Example":
        mqtt_example()

if __name__ == "__main__":
    main()
