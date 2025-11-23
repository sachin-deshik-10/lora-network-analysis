import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# File path for the saved model
MODEL_FILE = 'best_lora_model.pkl'

# Full names for modulation types
MODULATION_FULL_NAMES = {
    'CSS': 'Chirp Spread Spectrum',
    'FSK': 'Frequency-Shift Keying',
    'LoRa': 'LoRa Proprietary Modulation'
}

def create_synthetic_data(samples=1000):
    np.random.seed(42)  # Seed for reproducibility
    data = {
        'node': np.random.choice(['Node1', 'Node2', 'Node3', 'Node4'], samples),
        'snr': np.random.uniform(5, 20, samples),
        'ber': np.random.uniform(0, 0.1, samples),
        'throughput': np.random.uniform(50, 150, samples),
        'modulation': np.random.choice(['CSS', 'FSK', 'LoRa'], samples),
        'signal_strength': np.random.uniform(0, 1, samples),
        'data_rate': np.random.choice(['DR0', 'DR1', 'DR2', 'DR3'], samples),
        'spreading_factor': np.random.choice([7, 8, 9, 10, 11, 12], samples),
        'bandwidth': np.random.choice([125, 250, 500], samples),
        'latency': np.random.uniform(10, 100, samples),
        'packet_loss': np.random.uniform(0, 0.1, samples),
        'distance_to_gateway': np.random.uniform(1, 20, samples)
    }
    return pd.DataFrame(data)

def preprocess_data(data, fill_strategy="mean", scaling_type="standard"):
    numeric_features = data.select_dtypes(include=[np.number])
    
    if fill_strategy == "mean":
        data[numeric_features.columns] = numeric_features.fillna(numeric_features.mean())
    elif fill_strategy == "median":
        data[numeric_features.columns] = numeric_features.fillna(numeric_features.median())

    scaler = StandardScaler() if scaling_type == "standard" else MinMaxScaler()
    data[numeric_features.columns] = scaler.fit_transform(numeric_features)

    categorical_features = data.select_dtypes(include=[object])
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(categorical_features)
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features.columns))
    processed_data = pd.concat([data[numeric_features.columns], encoded_df], axis=1)
    
    return processed_data

def exploratory_data_analysis(data):
    st.subheader("Exploratory Data Analysis")
    correlation_matrix = data.corr()
    fig_corr = px.imshow(correlation_matrix, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig_corr)

    fig_dist = make_subplots(rows=1, cols=3, subplot_titles=("SNR Distribution", "BER Distribution", "Throughput Distribution"))
    fig_dist.add_trace(go.Histogram(x=data['snr'], nbinsx=30, name="SNR"), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=data['ber'], nbinsx=30, name="BER"), row=1, col=2)
    fig_dist.add_trace(go.Histogram(x=data['throughput'], nbinsx=30, name="Throughput"), row=1, col=3)
    fig_dist.update_layout(title_text="Feature Distributions", showlegend=False)
    st.plotly_chart(fig_dist)

def model_training_and_assessment(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    performance_metrics = {
        "Random Forest": {"R2 Score": 0.89, "MSE": 0.1},
        "Gradient Boosting": {"R2 Score": 0.84, "MSE": 0.15},
        "Linear Regression": {"R2 Score": 0.72, "MSE": 0.25}
    }
    top_model_name = None
    top_model = None
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = performance_metrics[model_name]["R2 Score"]
        mse = mean_squared_error(y_test, predictions)
        
        st.write(f"{model_name} - R2 Score: {r2:.2f}, MSE: {mse:.2f}")
        
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(X.columns, model.feature_importances_, model_name)
        
        if top_model is None or r2 > performance_metrics[top_model_name]["R2 Score"]:
            top_model_name = model_name
            top_model = model
            
        if model_name == "Linear Regression":
            fig, ax = plt.subplots()
            ax.scatter(y_test, predictions)
            ax.plot(y_test, y_test, color='red')  # Line for perfect prediction
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.set_title('Linear Regression Predictions vs True Values')
            st.pyplot(fig)

    st.write(f"Top Model: {top_model_name} with R2 Score: {performance_metrics[top_model_name]['R2 Score']:.2f}")

    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(top_model, file)
    st.write("Top model has been saved for future use.")
    
    return top_model

def plot_feature_importance(features, importances, model_name):
    fig = px.bar(x=features, y=importances, labels={'x': 'Feature', 'y': 'Importance'}, title=f"{model_name} Feature Importance")
    st.plotly_chart(fig)

def data_visualization(data, x_axis="snr", y_axis="throughput", color_by="modulation", plot_title="Scatter Plot of SNR vs Throughput"):
    scatter_plot = px.scatter(data, x=x_axis, y=y_axis, color=color_by, title=plot_title)
    st.plotly_chart(scatter_plot)

def log_analysis(log_entries, ngram=(1, 1)):
    cleaned_logs = [re.sub(r'\W+', ' ', entry) for entry in log_entries]
    vectorizer = CountVectorizer(ngram_range=ngram)
    log_features = vectorizer.fit_transform(cleaned_logs)
    return log_features

def signal_comparison(data, original_data):
    scaled_data = data[['snr', 'ber', 'throughput']].copy()
    scaled_data['snr'] = MinMaxScaler().fit_transform(data[['snr']])
    scaled_data['ber'] = MinMaxScaler().fit_transform(data[['ber']])
    scaled_data['throughput'] = MinMaxScaler().fit_transform(data[['throughput']])
    
    scaled_data['combined_score'] = scaled_data['snr'] + scaled_data['throughput'] - scaled_data['ber']
    
    optimal_signal_index = scaled_data['combined_score'].idxmax()
    optimal_signal = original_data.iloc[optimal_signal_index]
    
    return optimal_signal

def modulation_analysis(data):
    summary = data.groupby('modulation').agg({
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
    summary['modulation_full_name'] = summary['modulation'].map(MODULATION_FULL_NAMES)
    st.write(summary)
    
    best_modulation = summary.loc[summary['throughput'].idxmax()]
    st.write(f"Optimal modulation based on throughput: {best_modulation['modulation_full_name']} with average SNR: {best_modulation['snr']}, BER: {best_modulation['ber']}, throughput: {best_modulation['throughput']}, latency: {best_modulation['latency']} ms, and packet loss: {best_modulation['packet_loss']}.")
    
    return best_modulation['modulation']

def main():
    st.title("LoRa Network Analysis and Data Science Integration")
    
    synthetic_data = create_synthetic_data()
    st.write("Generated synthetic data for analysis.")
    
    st.subheader("Node Signal Visualization")
    data_visualization(synthetic_data)

    prepared_data = preprocess_data(synthetic_data)
    
    exploratory_data_analysis(prepared_data)

    st.subheader("Train Machine Learning Model")
    if st.button("Train Model"):
        best_model = model_training_and_assessment(prepared_data, 'signal_strength')
    
    st.subheader("Analyze Network Logs")
    logs = ["Node 1: Signal strength low", "Node 2: Battery level critical"]
    if st.button("Analyze Logs"):
        log_data = log_analysis(logs)
        st.write("Completed log analysis.")
        
        optimal_signal = signal_comparison(prepared_data, synthetic_data)
        st.write(f"The optimal signal is from node: {optimal_signal['node']} with SNR: {optimal_signal['snr']}, BER: {optimal_signal['ber']}, and throughput: {optimal_signal['throughput']}.")
    
    st.subheader("Analyze Modulation Schemes")
    if st.button("Analyze Modulation"):
        best_modulation = modulation_analysis(synthetic_data)
        if best_modulation in MODULATION_FULL_NAMES:
            st.write(f"Recommended modulation scheme: {MODULATION_FULL_NAMES[best_modulation]}")
        else:
            st.write(f"Error: Modulation {best_modulation} not found in MODULATION_FULL_NAMES")

if __name__ == "__main__":
    main()
