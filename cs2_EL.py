import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def simulate_energy_consumption(packet_size, snr, modulation):
    # Assume an energy model based on modulation and SNR
    energy_per_bit = {"Chirp Modulation": 0.5, "Amplitude Modulation": 0.6, "Pulse Modulation": 0.4, "Frequency Modulation": 0.7}
    energy = packet_size * energy_per_bit[modulation] / snr
    return energy

def update_node_location(nodes_df, time_step):
    # Simple mobility model for nodes
    nodes_df['x'] += np.random.uniform(-1, 1, size=len(nodes_df))
    nodes_df['y'] += np.random.uniform(-1, 1, size=len(nodes_df))
    return nodes_df


# Function to load data
def load_data():
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
            return (dataframes["nodes"], dataframes["distances"], dataframes["obstacles"],
                    dataframes["pressure"], dataframes["humidity"], dataframes["temperature"])
        except Exception as e:
            st.error(f"Error loading files: {e}")
            return None
    else:
        st.warning("Please upload all required files.")
        return None

def simulate_parameters_for_modulation(modulation):
    if modulation == "Chirp Modulation":
        snr = np.random.uniform(5, 15)
        ber = np.random.uniform(0.01, 0.1)
        correlation = np.random.uniform(0.8, 1.0)
        throughput = np.random.uniform(100, 200)  # kbps
        latency = np.random.uniform(5, 15)  # ms
    elif modulation == "Amplitude Modulation":
        snr = np.random.uniform(10, 20)
        ber = np.random.uniform(0.02, 0.12)
        correlation = np.random.uniform(0.75, 0.95)
        throughput = np.random.uniform(50, 150)  # kbps
        latency = np.random.uniform(10, 20)  # ms
    elif modulation == "Pulse Modulation":
        snr = np.random.uniform(8, 18)
        ber = np.random.uniform(0.015, 0.105)
        correlation = np.random.uniform(0.78, 0.98)
        throughput = np.random.uniform(80, 160)  # kbps
        latency = np.random.uniform(8, 18)  # ms
    elif modulation == "Frequency Modulation":
        snr = np.random.uniform(12, 22)
        ber = np.random.uniform(0.005, 0.085)
        correlation = np.random.uniform(0.85, 1.0)
        throughput = np.random.uniform(120, 220)  # kbps
        latency = np.random.uniform(3, 12)  # ms
    else:
        snr, ber, correlation, throughput, latency = 0, 0, 0, 0, 0
    return snr, ber, correlation, throughput, latency

def simulate_protocol_performance(protocol):
    if protocol == "ALOHA":
        success_rate = np.random.uniform(0.1, 0.5)
        throughput = success_rate * 100  # Random throughput based on success rate
        latency = np.random.uniform(20, 50)  # ms
    elif protocol == "CSMA/CD":
        success_rate = np.random.uniform(0.3, 0.8)
        throughput = success_rate * 150  # Random throughput based on success rate
        latency = np.random.uniform(10, 30)  # ms
    elif protocol == "TDMA":
        success_rate = np.random.uniform(0.5, 1.0)
        throughput = success_rate * 200  # Random throughput based on success rate
        latency = np.random.uniform(5, 20)  # ms
    else:
        success_rate, throughput, latency = 0, 0, 0
    return success_rate, throughput, latency

def plot_parameters(snr_values, ber_values, correlation_values, throughput_values, latency_values, modulation_type):
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))

    axs[0].plot(range(len(snr_values)), snr_values, marker='o')
    axs[0].set_title(f"SNR for {modulation_type}")
    axs[0].set_xlabel("Packet Number")
    axs[0].set_ylabel("SNR (dB)")

    axs[1].plot(range(len(ber_values)), ber_values, marker='o', color='r')
    axs[1].set_title(f"BER for {modulation_type}")
    axs[1].set_xlabel("Packet Number")
    axs[1].set_ylabel("BER")

    axs[2].plot(range(len(correlation_values)), correlation_values, marker='o', color='g')
    axs[2].set_title(f"Signal Correlation for {modulation_type}")
    axs[2].set_xlabel("Packet Number")
    axs[2].set_ylabel("Correlation")

    axs[3].plot(range(len(throughput_values)), throughput_values, marker='o', color='orange')
    axs[3].set_title(f"Throughput for {modulation_type}")
    axs[3].set_xlabel("Packet Number")
    axs[3].set_ylabel("Throughput (kbps)")

    axs[4].plot(range(len(latency_values)), latency_values, marker='o', color='purple')
    axs[4].set_title(f"Latency for {modulation_type}")
    axs[4].set_xlabel("Packet Number")
    axs[4].set_ylabel("Latency (ms)")

    for ax in axs:
        ax.grid(True)

    st.pyplot(fig)

def run_modulation_simulation(modulation, params):
    st.write(f"Running {modulation} Simulation...")
    for key, value in params.items():
        st.write(f"{key}: {value}")
    snr_values, ber_values, correlation_values, throughput_values, latency_values = [], [], [], [], []

    for _ in range(10):
        snr, ber, correlation, throughput, latency = simulate_parameters_for_modulation(modulation)
        snr_values.append(snr)
        ber_values.append(ber)
        correlation_values.append(correlation)
        throughput_values.append(throughput)
        latency_values.append(latency)

    plot_parameters(snr_values, ber_values, correlation_values, throughput_values, latency_values, modulation)

def plot_protocol_comparison(success_rates, throughputs, latencies, protocols):
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    # Plot Success Rate
    axs[0].bar(protocols, success_rates, color='blue')
    axs[0].set_title("Success Rate Across Protocols")
    axs[0].set_xlabel("Protocols")
    axs[0].set_ylabel("Success Rate")

    # Plot Throughput
    axs[1].bar(protocols, throughputs, color='green')
    axs[1].set_title("Throughput Across Protocols")
    axs[1].set_xlabel("Protocols")
    axs[1].set_ylabel("Throughput (kbps)")

    # Plot Latency
    axs[2].bar(protocols, latencies, color='red')
    axs[2].set_title("Latency Across Protocols")
    axs[2].set_xlabel("Protocols")
    axs[2].set_ylabel("Latency (ms)")

    for ax in axs:
        ax.grid(True)

    st.pyplot(fig)

def run_protocol_comparison(num_simulations, protocols):
    st.write("Running Protocol Comparison...")
    success_rates = []
    throughputs = []
    latencies = []

    for protocol in protocols:
        protocol_success_rate, protocol_throughput, protocol_latency = simulate_protocol_performance(protocol)
        success_rates.append(protocol_success_rate)
        throughputs.append(protocol_throughput)
        latencies.append(protocol_latency)

    plot_protocol_comparison(success_rates, throughputs, latencies, protocols)

def run_simulation_with_interference(nodes_df, distances_df, obstacles_df, pressure_df, humidity_df, temperature_df, num_packets, interference_level):
    st.write("Running Communication Network Simulation with Interference...")
    snr_values, ber_values, correlation_values, throughput_values, latency_values = [], [], [], [], []

    for _ in range(num_packets):
        # Simulating the effects of interference
        snr, ber, correlation, throughput, latency = simulate_parameters_for_modulation("Chirp Modulation")
        # Adjusting the SNR and BER based on interference level
        snr -= interference_level * np.random.uniform(0.1, 0.5)
        ber += interference_level * np.random.uniform(0.001, 0.005)
        snr_values.append(snr)
        ber_values.append(ber)
        correlation_values.append(correlation)
        throughput_values.append(throughput)
        latency_values.append(latency)

    plot_parameters(snr_values, ber_values, correlation_values, throughput_values, latency_values, "Communication with Interference")

def run_modulation_comparison(nodes_df, distances_df, obstacles_df, pressure_df, humidity_df, temperature_df, num_packets, interference_level, modulations):
    st.write("Running Modulation Comparison...")
    snr_results = {mod: [] for mod in modulations}
    ber_results = {mod: [] for mod in modulations}
    correlation_results = {mod: [] for mod in modulations}
    throughput_results = {mod: [] for mod in modulations}
    latency_results = {mod: [] for mod in modulations}

    for modulation in modulations:
        for _ in range(num_packets):
            snr, ber, correlation, throughput, latency = simulate_parameters_for_modulation(modulation)
            # Adjusting for interference
            snr -= interference_level * np.random.uniform(0.1, 0.5)
            ber += interference_level * np.random.uniform(0.001, 0.005)
            snr_results[modulation].append(snr)
            ber_results[modulation].append(ber)
            correlation_results[modulation].append(correlation)
            throughput_results[modulation].append(throughput)
            latency_results[modulation].append(latency)

    # Plotting comparison results
    plot_comparison(snr_results, ber_results, correlation_results, throughput_results, latency_results)

def plot_comparison(snr_results, ber_results, correlation_results, throughput_results, latency_results):
    fig, axs = plt.subplots(5, 1, figsize=(10, 25))

    # Plot SNR
    for modulation, snr_values in snr_results.items():
        axs[0].plot(range(len(snr_values)), snr_values, label=modulation, marker='o')
    axs[0].set_title("SNR Comparison Across Modulation Schemes")
    axs[0].set_xlabel("Packet Number")
    axs[0].set_ylabel("SNR (dB)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot BER
    for modulation, ber_values in ber_results.items():
        axs[1].plot(range(len(ber_values)), ber_values, label=modulation, marker='o', linestyle='--')
    axs[1].set_title("BER Comparison Across Modulation Schemes")
    axs[1].set_xlabel("Packet Number")
    axs[1].set_ylabel("BER")
    axs[1].legend()
    axs[1].grid(True)

    # Plot Correlation
    for modulation, correlation_values in correlation_results.items():
        axs[2].plot(range(len(correlation_values)), correlation_values, label=modulation, marker='o', linestyle='-.')
    axs[2].set_title("Signal Correlation Comparison Across Modulation Schemes")
    axs[2].set_xlabel("Packet Number")
    axs[2].set_ylabel("Correlation")
    axs[2].legend()
    axs[2].grid(True)

    # Plot Throughput
    for modulation, throughput_values in throughput_results.items():
        axs[3].plot(range(len(throughput_values)), throughput_values, label=modulation, marker='o', linestyle=':')
    axs[3].set_title("Throughput Comparison Across Modulation Schemes")
    axs[3].set_xlabel("Packet Number")
    axs[3].set_ylabel("Throughput (kbps)")
    axs[3].legend()
    axs[3].grid(True)

    # Plot Latency
    for modulation, latency_values in latency_results.items():
        axs[4].plot(range(len(latency_values)), latency_values, label=modulation, marker='o', linestyle='--')
    axs[4].set_title("Latency Comparison Across Modulation Schemes")
    axs[4].set_xlabel("Packet Number")
    axs[4].set_ylabel("Latency (ms)")
    axs[4].legend()
    axs[4].grid(True)

    st.pyplot(fig)

def main():
    st.sidebar.title("LoRa Simulation")
    simulation_option = st.sidebar.selectbox("Select Simulation Option", [
        "Chirp Modulation",
        "Amplitude Modulation",
        "Pulse Modulation",
        "Frequency Modulation",
        "Communication Network Simulation",
        "Compare Modulations",
        "Protocol Comparison"
    ])

    if simulation_option in ["Chirp Modulation", "Amplitude Modulation", "Pulse Modulation", "Frequency Modulation"]:
        st.title(f"{simulation_option} Simulation")
        params = {
            "Frequency": st.number_input("Frequency (Hz)", min_value=10.0, max_value=10e6, value=1e6, step=1e3),
            "Depth/Width": st.number_input("Depth/Width", min_value=0.0, value=0.5, step=0.1)
        }
        if st.button(f"Run {simulation_option} Simulation"):
            run_modulation_simulation(simulation_option, params)

    elif simulation_option == "Communication Network Simulation":
        st.title("Communication Network Simulation")
        data = load_data()
        if data:
            nodes_df, distances_df, obstacles_df, pressure_df, humidity_df, temperature_df = data
            num_packets = st.number_input("Number of Packets", min_value=1, value=100, step=1)
            interference_level = st.number_input("Interference Level", min_value=0.0, value=5.0, step=0.1)
            if st.button("Run Communication Network Simulation"):
                run_simulation_with_interference(
                    nodes_df, distances_df, obstacles_df, pressure_df, humidity_df, temperature_df, num_packets, interference_level
                )

    elif simulation_option == "Compare Modulations":
        st.title("Compare Modulation Schemes")
        modulations = st.multiselect("Select Modulation Schemes", [
            "Chirp Modulation",
            "Amplitude Modulation",
            "Pulse Modulation",
            "Frequency Modulation"
        ])
        if modulations:
            num_packets = st.number_input("Number of Packets", min_value=1, value=100, step=1)
            interference_level = st.number_input("Interference Level", min_value=0.0, value=5.0, step=0.1)
            data = load_data()
            if data:
                nodes_df, distances_df, obstacles_df, pressure_df, humidity_df, temperature_df = data
                if st.button("Run Modulation Comparison"):
                    run_modulation_comparison(
                        nodes_df, distances_df, obstacles_df, pressure_df, humidity_df, temperature_df, num_packets, interference_level, modulations
                    )

    elif simulation_option == "Protocol Comparison":
        st.title("Compare Communication Protocols")
        protocols = ["ALOHA", "CSMA/CD", "TDMA"]
        num_simulations = st.number_input("Number of Simulations", min_value=1, value=10, step=1)
        if st.button("Run Protocol Comparison"):
            run_protocol_comparison(num_simulations, protocols)

if __name__ == "__main__":
    main()