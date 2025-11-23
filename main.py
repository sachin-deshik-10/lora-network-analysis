import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime, timedelta
import time

# File path for the saved model
MODEL_FILE = 'best_lora_model.pkl'

# Configure page for wide layout
st.set_page_config(
    page_title="LoRa Network Analysis & Optimization Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Full names for modulation types
MODULATION_FULL_NAMES = {
    'CSS': 'Chirp Spread Spectrum',
    'FSK': 'Frequency-Shift Keying',
    'LoRa': 'LoRa Proprietary Modulation'
}

def create_synthetic_data(samples=1000, real_time=False):
    """
    Create synthetic LoRa network data with realistic parameters.
    If real_time=True, adds timestamp for real-time simulation.
    """
    if not real_time:
        np.random.seed(42)  # Seed for reproducibility
    else:
        np.random.seed(int(time.time()) % 1000)  # Dynamic seed for real-time
    
    # Generate timestamps for real-time simulation
    if real_time:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=samples)
    
    data = {
        'node': np.random.choice(['Node1', 'Node2', 'Node3', 'Node4'], samples),
        'snr': np.random.uniform(5, 20, samples),
        'ber': np.random.uniform(0.01, 0.1, samples),  # Bit Error Rate
        'throughput': np.random.uniform(50, 150, samples),  # kbps
        'modulation': np.random.choice(['CSS', 'FSK', 'LoRa'], samples),
        'signal_strength': np.random.uniform(0.1, 1, samples),  # Signal strength ratio
        'data_rate': np.random.choice(['DR0', 'DR1', 'DR2', 'DR3'], samples),
        'spreading_factor': np.random.choice([7, 8, 9, 10, 11, 12], samples),
        'bandwidth': np.random.choice([125, 250, 500], samples),  # kHz
        'latency': np.random.uniform(10, 100, samples),  # ms
        'packet_loss': np.random.uniform(0.01, 0.1, samples),  # Packet loss rate
        'distance_to_gateway': np.random.uniform(1, 20, samples)  # km
    }
    
    if real_time:
        data['timestamp'] = timestamps
    
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
    """
    Comprehensive EDA with enhanced visualizations and proper labeling.
    """
    st.subheader("Exploratory Data Analysis")
    st.write("Network parameter correlation and distribution analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation matrix with enhanced labels
        st.write("#### Feature Correlation Matrix")
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            title="Network Parameter Correlation Analysis",
            labels=dict(
                x="Network Parameters",
                y="Network Parameters",
                color="Correlation Coefficient"
            ),
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig_corr.update_layout(
            width=600,
            height=500,
            font=dict(size=10)
        )
        st.plotly_chart(fig_corr, width='stretch')
    
    with col2:
        # Key metrics display
        st.write("#### Key Network Metrics")
        metrics_df = pd.DataFrame({
            'Metric': [
                'Average SNR',
                'Average BER',
                'Average Throughput',
                'Average Latency',
                'Packet Loss Rate'
            ],
            'Value': [
                f"{data['snr'].mean():.2f} dB",
                f"{data['ber'].mean():.4f}",
                f"{data['throughput'].mean():.2f} kbps",
                f"{data['latency'].mean():.2f} ms",
                f"{data['packet_loss'].mean():.4f}"
            ]
        })
        st.dataframe(metrics_df, width='stretch', hide_index=True)
        
        # Network health indicator
        health_score = (
            (data['snr'].mean() / 20) * 0.4 +
            (1 - data['ber'].mean() / 0.1) * 0.3 +
            (data['throughput'].mean() / 150) * 0.3
        )
        
        health_status = "Optimal" if health_score > 0.7 else "Attention Required"
        st.metric(
            "Network Health Score",
            f"{health_score * 100:.1f}%",
            delta=health_status
        )

    # Enhanced distribution plots with proper labels
    st.write("#### Network Parameter Distributions")
    fig_dist = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "SNR Distribution (dB)",
            "Bit Error Rate (BER)",
            "Throughput (kbps)",
            "Latency Distribution (ms)",
            "Packet Loss Rate",
            "Distance to Gateway (km)"
        )
    )
    
    fig_dist.add_trace(
        go.Histogram(
            x=data['snr'], nbinsx=30, name="SNR",
            marker_color='#1f77b4'
        ),
        row=1, col=1
    )
    fig_dist.add_trace(
        go.Histogram(
            x=data['ber'], nbinsx=30, name="BER",
            marker_color='#ff7f0e'
        ),
        row=1, col=2
    )
    fig_dist.add_trace(
        go.Histogram(
            x=data['throughput'], nbinsx=30, name="Throughput",
            marker_color='#2ca02c'
        ),
        row=1, col=3
    )
    fig_dist.add_trace(
        go.Histogram(
            x=data['latency'], nbinsx=30, name="Latency",
            marker_color='#d62728'
        ),
        row=2, col=1
    )
    fig_dist.add_trace(
        go.Histogram(
            x=data['packet_loss'], nbinsx=30, name="Packet Loss",
            marker_color='#9467bd'
        ),
        row=2, col=2
    )
    fig_dist.add_trace(
        go.Histogram(
            x=data['distance_to_gateway'], nbinsx=30, name="Distance",
            marker_color='#8c564b'
        ),
        row=2, col=3
    )
    
    fig_dist.update_xaxes(title_text="SNR (dB)", row=1, col=1)
    fig_dist.update_xaxes(title_text="BER", row=1, col=2)
    fig_dist.update_xaxes(title_text="Throughput (kbps)", row=1, col=3)
    fig_dist.update_xaxes(title_text="Latency (ms)", row=2, col=1)
    fig_dist.update_xaxes(title_text="Packet Loss Rate", row=2, col=2)
    fig_dist.update_xaxes(title_text="Distance (km)", row=2, col=3)
    
    fig_dist.update_yaxes(title_text="Frequency", row=1, col=1)
    fig_dist.update_yaxes(title_text="Frequency", row=1, col=2)
    fig_dist.update_yaxes(title_text="Frequency", row=1, col=3)
    fig_dist.update_yaxes(title_text="Frequency", row=2, col=1)
    fig_dist.update_yaxes(title_text="Frequency", row=2, col=2)
    fig_dist.update_yaxes(title_text="Frequency", row=2, col=3)
    
    fig_dist.update_layout(
        title_text="Network Parameter Distribution Analysis",
        showlegend=False,
        height=600,
        font=dict(size=10)
    )
    st.plotly_chart(fig_dist, width='stretch')


def model_training_and_assessment(data, target_column):
    """
    Train and evaluate multiple ML models with comprehensive metrics.
    """
    st.subheader("Machine Learning Model Training & Evaluation")
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    performance_results = []
    top_model_name = None
    top_model = None
    best_r2 = -np.inf
    
    # Create columns for side-by-side comparison
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    for idx, (model_name, model) in enumerate(models.items()):
        with cols[idx]:
            st.write(f"#### {model_name}")
            
            # Train model
            with st.spinner(f'Training {model_name}...'):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Calculate comprehensive metrics
                r2 = r2_score(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, predictions)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=5, 
                    scoring='r2'
                )
                
                # Display metrics
                st.metric("R² Score", f"{r2:.4f}")
                st.metric("RMSE", f"{rmse:.4f}")
                st.metric("MAE", f"{mae:.4f}")
                st.metric("CV Score (mean)", f"{cv_scores.mean():.4f}")
                
                performance_results.append({
                    'Model': model_name,
                    'R² Score': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'CV Score': cv_scores.mean()
                })
                
                # Track best model
                if r2 > best_r2:
                    best_r2 = r2
                    top_model_name = model_name
                    top_model = model
                
                # Plot predictions vs actual for Linear Regression
                if model_name == "Linear Regression":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_test, 
                        y=predictions,
                        mode='markers',
                        name='Predictions',
                        marker=dict(size=8, opacity=0.6, color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title=f"{model_name}: Predictions vs Actual",
                        xaxis_title="Actual Values (Signal Strength)",
                        yaxis_title="Predicted Values (Signal Strength)",
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, width='stretch')
    
    # Display comparison table
    st.write("#### Model Performance Comparison")
    results_df = pd.DataFrame(performance_results)
    st.dataframe(results_df, width='stretch', hide_index=True)
    
    # Highlight best model
    st.success(
        f"🏆 **Best Model: {top_model_name}** | "
        f"R² Score: {best_r2:.4f} | "
        f"Reason: Highest R² score indicates best fit to data"
    )
    
    # Feature importance for tree-based models
    if hasattr(top_model, 'feature_importances_'):
        st.write(f"#### Feature Importance - {top_model_name}")
        plot_feature_importance(X.columns, top_model.feature_importances_, top_model_name)

    # Save the top model
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(top_model, file)
    st.info(f"✅ Model '{top_model_name}' saved successfully to {MODEL_FILE}")
    
    return top_model

def plot_feature_importance(features, importances, model_name):
    """
    Plot feature importance with proper labels.
    """
    # Create dataframe and sort by importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=importance_df['Importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance Score")
        )
    ))
    
    fig.update_layout(
        title=f"{model_name} - Feature Importance Analysis",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, width='stretch')

def data_visualization(data):
    """
    Enhanced 3D visualization with proper axis labels and annotations.
    """
    st.subheader("📈 Advanced Network Visualization")
    st.write("Explore multi-dimensional relationships in LoRa network parameters.")
    
    # 3D Scatter Plot with enhanced labels
    fig = px.scatter_3d(
        data, 
        x='snr', 
        y='throughput', 
        z='ber', 
        color='modulation',
        size='latency', 
        animation_frame='spreading_factor' if 'spreading_factor' in data.columns else None,
        hover_data=['node', 'distance_to_gateway', 'packet_loss'],
        title='3D Network Performance Analysis: SNR vs Throughput vs BER',
        labels={
            'snr': 'Signal-to-Noise Ratio (dB)',
            'throughput': 'Throughput (kbps)',
            'ber': 'Bit Error Rate',
            'modulation': 'Modulation Scheme',
            'latency': 'Latency (ms)',
            'spreading_factor': 'Spreading Factor (SF)'
        },
        opacity=0.7,
        color_discrete_map={
            'CSS': '#1f77b4',
            'FSK': '#ff7f0e',
            'LoRa': '#2ca02c'
        }
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Signal-to-Noise Ratio (dB)',
            yaxis_title='Throughput (kbps)',
            zaxis_title='Bit Error Rate (BER)',
            xaxis=dict(backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(backgroundcolor="rgb(230, 230,230)")
        ),
        height=700,
        font=dict(size=11)
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Real-time monitoring simulation
    st.write("#### Real-Time Network Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SNR vs Distance scatter plot
        fig_snr_dist = px.scatter(
            data,
            x='distance_to_gateway',
            y='snr',
            color='modulation',
            size='signal_strength',
            title='SNR vs Distance to Gateway',
            labels={
                'distance_to_gateway': 'Distance to Gateway (km)',
                'snr': 'Signal-to-Noise Ratio (dB)',
                'modulation': 'Modulation Type',
                'signal_strength': 'Signal Strength'
            },
            hover_data=['node', 'throughput']
        )
        fig_snr_dist.update_layout(height=400)
        st.plotly_chart(fig_snr_dist, width='stretch')
    
    with col2:
        # Throughput vs Latency
        fig_tp_lat = px.scatter(
            data,
            x='latency',
            y='throughput',
            color='modulation',
            size='packet_loss',
            title='Throughput vs Latency Analysis',
            labels={
                'latency': 'Latency (ms)',
                'throughput': 'Throughput (kbps)',
                'modulation': 'Modulation Type',
                'packet_loss': 'Packet Loss Rate'
            },
            hover_data=['node', 'snr']
        )
        fig_tp_lat.update_layout(height=400)
        st.plotly_chart(fig_tp_lat, width='stretch')
    
    # Time series simulation if timestamp exists
    if 'timestamp' in data.columns:
        st.write("#### Time Series Analysis")
        fig_ts = go.Figure()
        
        for node in data['node'].unique():
            node_data = data[data['node'] == node]
            fig_ts.add_trace(go.Scatter(
                x=node_data['timestamp'],
                y=node_data['throughput'],
                mode='lines+markers',
                name=node,
                line=dict(width=2)
            ))
        
        fig_ts.update_layout(
            title='Real-Time Throughput Monitoring by Node',
            xaxis_title='Time',
            yaxis_title='Throughput (kbps)',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_ts, width='stretch')


def modulation_analysis(data):
    """
    Comprehensive modulation scheme analysis with enhanced visualizations.
    """
    st.subheader("📡 Modulation Scheme Analysis")
    
    summary = data.groupby('modulation').agg({
        'snr': 'mean',
        'ber': 'mean',
        'throughput': 'mean',
        'data_rate': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A',
        'spreading_factor': 'mean',
        'bandwidth': 'mean',
        'latency': 'mean',
        'packet_loss': 'mean',
        'distance_to_gateway': 'mean'
    }).reset_index()
    
    summary['modulation_full_name'] = summary['modulation'].map(
        MODULATION_FULL_NAMES
    )
    
    # Display comprehensive modulation parameters
    st.write("#### Modulation Performance Metrics")
    summary_display = summary.copy()
    summary_display.columns = [
        "Modulation Type", "Avg SNR (dB)", "Avg BER", "Avg Throughput (kbps)",
        "Common Data Rate", "Avg Spreading Factor", "Avg Bandwidth (kHz)",
        "Avg Latency (ms)", "Avg Packet Loss", "Avg Distance (km)",
        "Full Modulation Name"
    ]
    
    # Round numerical columns
    numeric_cols = summary_display.select_dtypes(include=[np.number]).columns
    summary_display[numeric_cols] = summary_display[numeric_cols].round(3)
    
    st.dataframe(summary_display, width='stretch', hide_index=True)

    # Comparative bar charts
    st.write("#### Comparative Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Throughput comparison
        fig_throughput = px.bar(
            summary,
            x='modulation_full_name',
            y='throughput',
            color='modulation_full_name',
            title='Average Throughput by Modulation Scheme',
            labels={
                'modulation_full_name': 'Modulation Type',
                'throughput': 'Average Throughput (kbps)'
            },
            text_auto='.2f'
        )
        fig_throughput.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_throughput, width='stretch')
    
    with col2:
        # Latency comparison
        fig_latency = px.bar(
            summary,
            x='modulation_full_name',
            y='latency',
            color='modulation_full_name',
            title='Average Latency by Modulation Scheme',
            labels={
                'modulation_full_name': 'Modulation Type',
                'latency': 'Average Latency (ms)'
            },
            text_auto='.2f'
        )
        fig_latency.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_latency, width='stretch')

    # Identify and display the best modulation
    best_modulation = summary.loc[summary['throughput'].idxmax()]
    
    st.write("#### 🏆 Optimal Modulation Recommendation")
    
    # Create performance score
    normalized_throughput = (best_modulation['throughput'] - summary['throughput'].min()) / \
                           (summary['throughput'].max() - summary['throughput'].min())
    normalized_latency = 1 - ((best_modulation['latency'] - summary['latency'].min()) / \
                             (summary['latency'].max() - summary['latency'].min()))
    performance_score = (normalized_throughput * 0.6 + normalized_latency * 0.4) * 100
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(
            f"""
            <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; 
                        border-left: 5px solid #4caf50;">
                <h4 style="color: #2e7d32; margin-top: 0;">
                    Recommended: {best_modulation['modulation_full_name']}
                </h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>📊 <strong>Avg SNR:</strong> {best_modulation['snr']:.2f} dB</li>
                    <li>📉 <strong>Bit Error Rate:</strong> {best_modulation['ber']:.4f}</li>
                    <li>🚀 <strong>Throughput:</strong> {best_modulation['throughput']:.2f} kbps</li>
                    <li>⏱️ <strong>Latency:</strong> {best_modulation['latency']:.2f} ms</li>
                    <li>📦 <strong>Packet Loss:</strong> {best_modulation['packet_loss']:.4f}</li>
                    <li>📏 <strong>Avg Distance:</strong> {best_modulation['distance_to_gateway']:.2f} km</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )
    
    with col2:
        st.metric(
            "Performance Score",
            f"{performance_score:.1f}%",
            delta="Optimal",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Efficiency Rating",
            f"{best_modulation['throughput']/best_modulation['latency']:.2f}",
            delta="kbps/ms"
        )
    
    return best_modulation['modulation']


def log_analysis(log_entries):
    """
    Enhanced network log analysis with pattern detection.
    """
    st.subheader("📋 Network Logs Analysis")
    
    cleaned_logs = [re.sub(r'\W+', ' ', entry) for entry in log_entries]
    vectorizer = CountVectorizer(max_features=10)
    log_features = vectorizer.fit_transform(cleaned_logs)
    
    # Display log entries
    st.write("#### Recent Log Entries")
    for idx, log in enumerate(log_entries, 1):
        log_type = "⚠️ Warning" if "critical" in log.lower() or "low" in log.lower() else "ℹ️ Info"
        st.write(f"{log_type} - Entry {idx}: {log}")
    
    # Display extracted features
    feature_names = vectorizer.get_feature_names_out()
    st.write("#### Extracted Keywords")
    st.write(", ".join(feature_names))
    
    st.info("✅ Log analysis completed. Feature patterns extracted for monitoring.")


def calculate_efficiency(data):
    """
    Calculate and visualize network efficiency metrics.
    """
    st.subheader("⚡ Network Efficiency Analysis")
    
    # Calculate efficiency as throughput per unit latency
    data['efficiency'] = data['throughput'] / data['latency']
    avg_efficiency = data['efficiency'].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average Efficiency",
            f"{avg_efficiency:.3f}",
            delta="kbps/ms"
        )
    
    with col2:
        max_efficiency = data['efficiency'].max()
        st.metric(
            "Peak Efficiency",
            f"{max_efficiency:.3f}",
            delta=f"+{((max_efficiency/avg_efficiency - 1) * 100):.1f}%"
        )
    
    with col3:
        # Network utilization score
        utilization = (data['throughput'].mean() / 150) * 100
        st.metric(
            "Network Utilization",
            f"{utilization:.1f}%",
            delta="of capacity"
        )
    
    # Efficiency distribution
    fig_efficiency = px.histogram(
        data,
        x='efficiency',
        nbins=50,
        title='Network Efficiency Distribution',
        labels={
            'efficiency': 'Efficiency (kbps/ms)',
            'count': 'Frequency'
        },
        color_discrete_sequence=['#4caf50']
    )
    fig_efficiency.update_layout(
        xaxis_title='Efficiency (kbps/ms)',
        yaxis_title='Frequency',
        height=400
    )
    st.plotly_chart(fig_efficiency, width='stretch')
    
    return avg_efficiency


def main():
    """
    Main application with enhanced real-time monitoring capabilities.
    """
    st.title("📡 Advanced LoRa Network Analysis & Real-Time Monitoring")
    
    st.markdown("""
        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; 
                    margin-bottom: 20px; border-left: 5px solid #2196f3;">
            <h4 style="color: #1565c0; margin-top: 0;">
                🚀 Welcome to Advanced LoRa Network Analytics
            </h4>
            <p style="color: #424242; margin-bottom: 0;">
                Leverage data-driven insights and machine learning to optimize your 
                LoRa network performance. This platform provides real-time monitoring, 
                predictive analytics, and actionable recommendations for network optimization.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Real-time mode toggle
        real_time_mode = st.checkbox(
            "Enable Real-Time Simulation",
            value=False,
            help="Simulate real-time data updates with timestamps"
        )
        
        # Sample size selector
        sample_size = st.slider(
            "Number of Data Samples",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Select the number of data points to generate"
        )
        
        # Data preprocessing options
        st.subheader("Data Preprocessing")
        fill_strategy = st.selectbox(
            "Missing Value Strategy",
            ["mean", "median"],
            help="Strategy for handling missing values"
        )
        
        scaling_type = st.selectbox(
            "Scaling Method",
            ["standard", "minmax"],
            help="Feature scaling method"
        )
        
        # Auto-refresh for real-time mode
        if real_time_mode:
            auto_refresh = st.checkbox(
                "Auto-refresh (30s)",
                value=False,
                help="Automatically refresh data every 30 seconds"
            )
            if auto_refresh:
                time.sleep(30)
                st.rerun()
    
    # Generate synthetic data
    with st.spinner('Generating network data...'):
        synthetic_data = create_synthetic_data(
            samples=sample_size,
            real_time=real_time_mode
        )
    
    st.success(f"✅ Generated {len(synthetic_data)} data samples successfully!")
    
    # Display raw data summary
    with st.expander("📊 View Raw Data Summary", expanded=False):
        st.write("#### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(synthetic_data))
        with col2:
            st.metric("Number of Nodes", synthetic_data['node'].nunique())
        with col3:
            st.metric("Modulation Types", synthetic_data['modulation'].nunique())
        with col4:
            st.metric("Features", len(synthetic_data.columns))
        
        st.dataframe(synthetic_data.head(10), width='stretch')
        st.write(synthetic_data.describe())
    
    # Visualization section
    st.header("📈 Network Performance Visualization")
    data_visualization(synthetic_data)

    # Preprocess data
    with st.spinner('Preprocessing data...'):
        prepared_data = preprocess_data(
            synthetic_data.copy(),
            fill_strategy=fill_strategy,
            scaling_type=scaling_type
        )
    
    # EDA section
    st.header("🔍 Exploratory Data Analysis")
    exploratory_data_analysis(prepared_data)

    # Machine Learning section
    st.header("🤖 Machine Learning Model Training")
    
    with st.expander("ℹ️ About Model Training", expanded=False):
        st.write("""
            This section trains multiple regression models to predict signal strength 
            based on network parameters. Models are evaluated using:
            - **R² Score**: Measures how well the model fits the data (higher is better)
            - **RMSE**: Root Mean Squared Error (lower is better)
            - **MAE**: Mean Absolute Error (lower is better)
            - **CV Score**: Cross-validation score for model robustness
        """)
    
    if st.button("🚀 Train Models", type="primary", width='stretch'):
        with st.spinner('Training machine learning models...'):
            best_model = model_training_and_assessment(
                prepared_data,
                'signal_strength'
            )
    
    # Modulation analysis section
    st.header("📡 Modulation Scheme Analysis")
    
    with st.expander("ℹ️ About Modulation Analysis", expanded=False):
        st.write("""
            Compare different modulation schemes (CSS, FSK, LoRa) to identify 
            the optimal configuration for your network based on:
            - Throughput performance
            - Latency characteristics
            - Error rates and reliability
            - Distance capabilities
        """)
    
    if st.button("🔬 Analyze Modulations", type="primary", width='stretch'):
        with st.spinner('Analyzing modulation schemes...'):
            best_modulation = modulation_analysis(synthetic_data)
            st.balloons()

    # Network logs section
    st.header("📋 Network Logs & Diagnostics")
    
    # Simulate some logs
    sample_logs = [
        "Node1: Signal strength degraded - Distance: 15.2 km",
        "Node2: Battery level critical - Action required",
        "Node3: High packet loss detected - BER: 0.085",
        "Node4: Optimal performance - All metrics normal",
        "Gateway: Connection timeout on Node2"
    ]
    
    if st.button("📝 Analyze Logs", width='stretch'):
        log_analysis(sample_logs)
    
    # Efficiency calculation section
    st.header("⚡ Network Efficiency Metrics")
    efficiency = calculate_efficiency(synthetic_data)
    
    # Summary and recommendations
    st.header("📋 Summary & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
            **✅ Analysis Complete**
            
            Your LoRa network has been comprehensively analyzed. Key insights:
            - Network performance metrics calculated
            - Machine learning models trained for prediction
            - Optimal modulation scheme identified
            - Efficiency metrics computed
        """)
    
    with col2:
        st.warning("""
            **🎯 Next Steps**
            
            1. Review the optimal modulation recommendation
            2. Monitor nodes with low signal strength
            3. Address critical battery levels
            4. Consider gateway placement optimization
            5. Implement real-time monitoring alerts
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>🔬 Advanced LoRa Network Analysis Platform | 
            Built with Streamlit & Machine Learning</p>
            <p style="font-size: 0.9em;">
                Last updated: {} | Samples analyzed: {}
            </p>
        </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(synthetic_data)),
    unsafe_allow_html=True)


if __name__ == "__main__":
    main()
