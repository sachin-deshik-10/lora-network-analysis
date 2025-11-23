import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from cryptography.fernet import Fernet

# Generate a key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def calculate_energy_consumption(packet_size, snr, modulation, data_rate):
    energy_per_bit = {
        "Chirp Modulation": 0.5,
        "Amplitude Modulation": 0.6,
        "Pulse Modulation": 0.4,
        "Frequency Modulation": 0.7
    }
    energy = packet_size * energy_per_bit[modulation] / (snr * data_rate)
    return energy

def update_node_positions(nodes_df):
    if 'x' not in nodes_df.columns or 'y' not in nodes_df.columns:
        nodes_df['x'] = np.random.uniform(0, 100, size=len(nodes_df))
        nodes_df['y'] = np.random.uniform(0, 100, size=len(nodes_df))
    else:
        nodes_df['x'] += np.random.uniform(-1, 1, size=len(nodes_df))
        nodes_df['y'] += np.random.uniform(-1, 1, size=len(nodes_df))
    return nodes_df

def retrieve_data():
    files = {
        "nodes": st.file_uploader("Upload nodes configuration file", type=['xlsx']),
        "distances": st.file_uploader("Upload distance configuration file", type=['xlsx']),
        "obstacles": st.file_uploader("Upload obstacle configuration file", type=['xlsx']),
        "pressure": st.file_uploader("Upload pressure configuration file", type=['xlsx']),
        "humidity": st.file_uploader("Upload humidity configuration file", type=['xlsx']),
        "temperature": st.file_uploader("Upload temperature configuration file", type=['xlsx'])
    }

    if all(file is not None for file in files.values()):
        try:
            dataframes = {key: pd.read_excel(file) for key, file in files.items()}
            nodes_df = dataframes["nodes"]
            nodes_df = update_node_positions(nodes_df)
            return (
                nodes_df, dataframes["distances"], dataframes["obstacles"],
                dataframes["pressure"], dataframes["humidity"], dataframes["temperature"]
            )
        except Exception as e:
            st.error(f"Error loading files: {e}")
            return None
    else:
        st.warning("Please upload all required files.")
        return None

def generate_modulation_parameters(modulation, adr_enabled, frequency_unit, noise_level):
    data_rate = np.random.uniform(1, 10) if adr_enabled else 1
    snr_adjusted = np.random.uniform(5, 15) - noise_level

    if modulation == "Chirp Modulation":
        snr = snr_adjusted
        ber = np.random.uniform(0.01, 0.1)
        correlation = np.random.uniform(0.8, 1.0)
        throughput = np.random.uniform(100, 200) * data_rate
        latency = np.random.uniform(5, 15) / data_rate
    elif modulation == "Amplitude Modulation":
        snr = snr_adjusted + 5
        ber = np.random.uniform(0.02, 0.12)
        correlation = np.random.uniform(0.75, 0.95)
        throughput = np.random.uniform(50, 150) * data_rate
        latency = np.random.uniform(10, 20) / data_rate
    elif modulation == "Pulse Modulation":
        snr = snr_adjusted + 3
        ber = np.random.uniform(0.015, 0.105)
        correlation = np.random.uniform(0.78, 0.98)
        throughput = np.random.uniform(80, 160) * data_rate
        latency = np.random.uniform(8, 18) / data_rate
    elif modulation == "Frequency Modulation":
        snr = snr_adjusted + 7
        ber = np.random.uniform(0.005, 0.085)
        correlation = np.random.uniform(0.85, 1.0)
        throughput = np.random.uniform(120, 220) * data_rate
        latency = np.random.uniform(3, 12) / data_rate
    else:
        snr, ber, correlation, throughput, latency = 0, 0, 0, 0, 0

    return snr, ber, correlation, throughput, latency, data_rate

def simulate_battery_usage(initial_energy, energy_consumed, num_packets):
    battery_levels = [initial_energy - i * energy_consumed for i in range(num_packets)]
    return battery_levels

def secure_data(data):
    encrypted_data = cipher_suite.encrypt(data.encode())
    return encrypted_data

def unsecure_data(encrypted_data):
    decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
    return decrypted_data

def visualize_real_time_data(snr_values, ber_values, battery_levels, modulation_type):
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=snr_values, mode='lines+markers', name='SNR'))
    fig.add_trace(go.Scatter(y=ber_values, mode='lines+markers', name='BER', yaxis='y2'))
    fig.add_trace(go.Scatter(y=battery_levels, mode='lines+markers', name='Battery Level', yaxis='y3'))

    fig.update_layout(
        title=f"Real-Time Data for {modulation_type}",
        xaxis=dict(title='Time'),
        yaxis=dict(title='SNR (dB)'),
        yaxis2=dict(title='BER', overlaying='y', side='right'),
        yaxis3=dict(title='Battery Level', overlaying='y', side='left', position=0.85),
        legend=dict(x=0, y=1.2),
    )

    st.plotly_chart(fig)

def map_node_locations(nodes_df):
    fig = px.scatter(nodes_df, x='x', y='y', text=nodes_df.index, title="Node Locations")
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig)

def execute_modulation_simulation(modulation, params, nodes_df, adr_enabled, noise_level):
    st.write(f"Executing {modulation} Simulation with ADR {'enabled' if adr_enabled else 'disabled'}...")
    for key, value in params.items():
        st.write(f"{key}: {value}")

    snr_values, ber_values, battery_levels = [], [], []
    initial_energy = 1000
    packet_size = 256
    protocol_used = "LoRaWAN"

    for i in range(10):
        snr, ber, _, _, _, data_rate = generate_modulation_parameters(modulation, adr_enabled, params['Frequency Unit'], noise_level)
        energy_consumed = calculate_energy_consumption(packet_size, snr, modulation, data_rate)
        battery_levels.append(simulate_battery_usage(initial_energy, energy_consumed, 10))
        snr_values.append(snr)
        ber_values.append(ber)

        if i == 9:
            encrypted_data = secure_data(f"SNR: {snr}, BER: {ber}, Data Rate: {data_rate}, Protocol: {protocol_used}, Packet Type: Data, Packet Size: {packet_size} bytes")
            st.write(f"Final Encrypted Data: {encrypted_data}")

    visualize_real_time_data(snr_values, ber_values, battery_levels[-1], modulation)
    map_node_locations(nodes_df)

def main():
    st.sidebar.title("LoRa Communication Simulation")
    simulation_option = st.sidebar.selectbox("Select Simulation Type", [
        "Chirp Modulation",
        "Amplitude Modulation",
        "Pulse Modulation",
        "Frequency Modulation"
    ])

    adr_enabled = st.sidebar.checkbox("Enable Adaptive Data Rate (ADR)")
    noise_level = st.sidebar.slider("Noise Level", 0.0, 5.0, 0.0)

    data = retrieve_data()
    if data:
        nodes_df, _, _, _, _, _ = data

        if simulation_option in ["Chirp Modulation", "Amplitude Modulation", "Pulse Modulation", "Frequency Modulation"]:
            st.title(f"{simulation_option} Simulation")
            params = {
                "Frequency": st.number_input("Frequency", min_value=10.0, max_value=10e6, value=1e6, step=1e3),
                "Frequency Unit": st.selectbox("Frequency Unit", ["Hz", "kHz", "MHz", "GHz"]),
                "Depth/Width": st.number_input("Depth/Width", min_value=0.0, value=0.5, step=0.1)
            }

            if st.button(f"Start {simulation_option} Simulation"):
                execute_modulation_simulation(simulation_option, params, nodes_df, adr_enabled, noise_level)

if __name__ == "__main__":
    main()
