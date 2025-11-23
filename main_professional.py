"""
LoRa Network Analysis and Optimization Platform
Advanced networking insights and predictive analytics for LoRa WAN deployments
"""

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

# Configuration constants
MODEL_FILE = 'best_lora_model.pkl'
MODULATION_SCHEMES = {
    'CSS': 'Chirp Spread Spectrum',
    'FSK': 'Frequency-Shift Keying',
    'LoRa': 'LoRa Proprietary Modulation'
}

# Page configuration
st.set_page_config(
    page_title="LoRa Network Analysis & Optimization Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)


def generate_network_data(samples=1000, real_time_mode=False):
    """
    Generate realistic LoRa network telemetry data with physical correlations.
    
    Parameters:
    -----------
    samples : int
        Number of data points to generate
    real_time_mode : bool
        Enable timestamp generation for time-series analysis
        
    Returns:
    --------
    pd.DataFrame
        Synthetic network telemetry data with realistic correlations
    """
    if not real_time_mode:
        np.random.seed(42)
    else:
        np.random.seed(int(time.time()) % 1000)
    
    if real_time_mode:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        timestamps = pd.date_range(
            start=start_time,
            end=end_time,
            periods=samples
        )
    
    # Generate base parameters with realistic physical correlations
    gateway_distance_km = np.random.uniform(1, 20, samples)
    spreading_factor = np.random.choice([7, 8, 9, 10, 11, 12], samples)
    
    # SNR decreases with distance (path loss model: SNR = SNR0 - 10*n*log10(d))
    snr_base = 25 - 2.5 * np.log10(gateway_distance_km + 1)
    snr_db = snr_base + np.random.normal(0, 2, samples)
    snr_db = np.clip(snr_db, 5, 20)
    
    # Signal strength correlates with SNR
    signal_strength = 0.3 + (snr_db - 5) / 15 * 0.6
    signal_strength = signal_strength + np.random.normal(0, 0.05, samples)
    signal_strength = np.clip(signal_strength, 0.1, 1.0)
    
    # BER increases with lower SNR (exponential relationship)
    ber = 0.01 + 0.09 * np.exp(-(snr_db - 5) / 8)
    ber = ber + np.random.normal(0, 0.005, samples)
    ber = np.clip(ber, 0.01, 0.1)
    
    # Throughput depends on spreading factor and signal quality
    throughput_base = 150 - (spreading_factor - 7) * 15
    throughput_kbps = throughput_base * signal_strength
    throughput_kbps = throughput_kbps + np.random.normal(0, 5, samples)
    throughput_kbps = np.clip(throughput_kbps, 50, 150)
    
    # Latency increases with spreading factor and distance
    latency_ms = 10 + (spreading_factor - 7) * 8 + gateway_distance_km * 2
    latency_ms = latency_ms + np.random.normal(0, 5, samples)
    latency_ms = np.clip(latency_ms, 10, 100)
    
    # Packet loss correlates with BER
    packet_loss_rate = ber * 0.5 + np.random.uniform(0, 0.02, samples)
    packet_loss_rate = np.clip(packet_loss_rate, 0.01, 0.1)
    
    data = {
        'node_id': np.random.choice(
            ['Node1', 'Node2', 'Node3', 'Node4'],
            samples
        ),
        'snr_db': snr_db,
        'ber': ber,
        'throughput_kbps': throughput_kbps,
        'modulation_scheme': np.random.choice(
            ['CSS', 'FSK', 'LoRa'],
            samples
        ),
        'signal_strength': signal_strength,
        'data_rate': np.random.choice(
            ['DR0', 'DR1', 'DR2', 'DR3'],
            samples
        ),
        'spreading_factor': spreading_factor,
        'bandwidth_khz': np.random.choice([125, 250, 500], samples),
        'latency_ms': latency_ms,
        'packet_loss_rate': packet_loss_rate,
        'gateway_distance_km': gateway_distance_km
    }
    
    if real_time_mode:
        data['timestamp'] = timestamps
    
    return pd.DataFrame(data)


def preprocess_telemetry_data(
    data,
    imputation_strategy="mean",
    scaling_method="standard"
):
    """
    Preprocess network telemetry data for analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw network data
    imputation_strategy : str
        Strategy for missing value imputation ('mean' or 'median')
    scaling_method : str
        Feature scaling method ('standard' or 'minmax')
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed data ready for analysis
    """
    numeric_features = data.select_dtypes(include=[np.number])
    
    # Handle missing values
    if imputation_strategy == "mean":
        data[numeric_features.columns] = numeric_features.fillna(
            numeric_features.mean()
        )
    elif imputation_strategy == "median":
        data[numeric_features.columns] = numeric_features.fillna(
            numeric_features.median()
        )

    # Feature scaling
    if scaling_method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    data[numeric_features.columns] = scaler.fit_transform(numeric_features)

    # Encode categorical variables
    categorical_features = data.select_dtypes(include=[object])
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(categorical_features)
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(categorical_features.columns)
    )
    
    processed_data = pd.concat(
        [data[numeric_features.columns], encoded_df],
        axis=1
    )
    
    return processed_data


def perform_exploratory_analysis(data):
    """
    Conduct comprehensive exploratory data analysis on network telemetry.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Network telemetry data
    """
    st.subheader("Exploratory Data Analysis")
    st.write("Network parameter correlation and statistical distribution analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation analysis
        st.write("#### Network Parameter Correlation Matrix")
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            title="Inter-Parameter Correlation Analysis",
            labels=dict(
                x="Network Parameters",
                y="Network Parameters",
                color="Pearson Correlation Coefficient"
            ),
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
    fig_corr.update_layout(width=600, height=500, font=dict(size=10))
    st.plotly_chart(fig_corr, width='stretch')
    
    with col2:
        # Key performance indicators
        st.write("#### Key Performance Indicators (KPIs)")
        
        # Determine column names based on data structure
        snr_col = 'snr_db' if 'snr_db' in data.columns else 'snr'
        throughput_col = ('throughput_kbps' if 'throughput_kbps' 
                         in data.columns else 'throughput')
        latency_col = 'latency_ms' if 'latency_ms' in data.columns else 'latency'
        packet_loss_col = ('packet_loss_rate' if 'packet_loss_rate' 
                          in data.columns else 'packet_loss')
        
        metrics_df = pd.DataFrame({
            'KPI': [
                'Average SNR',
                'Average BER',
                'Average Throughput',
                'Average Latency',
                'Packet Loss Rate'
            ],
            'Value': [
                f"{data[snr_col].mean():.2f} dB",
                f"{data['ber'].mean():.4f}",
                f"{data[throughput_col].mean():.2f} kbps",
                f"{data[latency_col].mean():.2f} ms",
                f"{data[packet_loss_col].mean():.4f}"
            ]
        })
        st.dataframe(metrics_df, width='stretch', hide_index=True)
        
        # Network health assessment
        health_score = (
            (data[snr_col].mean() / 20) * 0.4 +
            (1 - data['ber'].mean() / 0.1) * 0.3 +
            (data[throughput_col].mean() / 150) * 0.3
        )
        
        health_status = "Optimal" if health_score > 0.7 else "Requires Attention"
        st.metric(
            "Network Health Index",
            f"{health_score * 100:.1f}%",
            delta=health_status
        )

    # Statistical distribution analysis
    st.write("#### Statistical Distribution of Network Parameters")
    
    fig_dist = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "SNR Distribution (dB)",
            "Bit Error Rate Distribution",
            "Throughput Distribution (kbps)",
            "Latency Distribution (ms)",
            "Packet Loss Rate Distribution",
            "Gateway Distance Distribution (km)"
        )
    )
    
    # Add histograms for each parameter
    dist_col = ('gateway_distance_km' if 'gateway_distance_km' 
               in data.columns else 'distance_to_gateway')
    
    fig_dist.add_trace(
        go.Histogram(
            x=data[snr_col], nbinsx=30, name="SNR",
            marker_color='#1f77b4', showlegend=False
        ),
        row=1, col=1
    )
    fig_dist.add_trace(
        go.Histogram(
            x=data['ber'], nbinsx=30, name="BER",
            marker_color='#ff7f0e', showlegend=False
        ),
        row=1, col=2
    )
    fig_dist.add_trace(
        go.Histogram(
            x=data[throughput_col], nbinsx=30, name="Throughput",
            marker_color='#2ca02c', showlegend=False
        ),
        row=1, col=3
    )
    fig_dist.add_trace(
        go.Histogram(
            x=data[latency_col], nbinsx=30, name="Latency",
            marker_color='#d62728', showlegend=False
        ),
        row=2, col=1
    )
    fig_dist.add_trace(
        go.Histogram(
            x=data[packet_loss_col], nbinsx=30, name="Packet Loss",
            marker_color='#9467bd', showlegend=False
        ),
        row=2, col=2
    )
    fig_dist.add_trace(
        go.Histogram(
            x=data[dist_col], nbinsx=30, name="Distance",
            marker_color='#8c564b', showlegend=False
        ),
        row=2, col=3
    )
    
    # Update axis labels
    fig_dist.update_xaxes(title_text="SNR (dB)", row=1, col=1)
    fig_dist.update_xaxes(title_text="Bit Error Rate", row=1, col=2)
    fig_dist.update_xaxes(title_text="Throughput (kbps)", row=1, col=3)
    fig_dist.update_xaxes(title_text="Latency (ms)", row=2, col=1)
    fig_dist.update_xaxes(title_text="Packet Loss Rate", row=2, col=2)
    fig_dist.update_xaxes(title_text="Distance (km)", row=2, col=3)
    
    for i in range(1, 3):
        for j in range(1, 4):
            fig_dist.update_yaxes(title_text="Frequency", row=i, col=j)
    
    fig_dist.update_layout(
        title_text="Network Parameter Statistical Distributions",
        showlegend=False,
        height=600,
        font=dict(size=10)
    )
    st.plotly_chart(fig_dist, width='stretch')


def train_predictive_models(data, target_variable):
    """
    Train and evaluate machine learning models for network prediction.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Preprocessed network data
    target_variable : str
        Target variable for prediction
        
    Returns:
    --------
    sklearn model
        Best performing model
    """
    st.subheader("Predictive Model Training & Evaluation")
    
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Optimized models for better performance and realistic predictions
    models = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        ),
        "Linear Regression": LinearRegression()
    }
    
    performance_results = []
    best_model = None
    best_model_name = None
    best_r2_score = -np.inf
    
    # Model evaluation
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    for idx, (model_name, model) in enumerate(models.items()):
        with cols[idx]:
            st.write(f"#### {model_name}")
            
            with st.spinner(f'Training {model_name}...'):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Calculate comprehensive performance metrics
                r2 = r2_score(y_test, predictions)
                r2_train = r2_score(y_train, model.predict(X_train))
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, predictions)
                mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
                
                # Cross-validation for robustness check
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=5, scoring='r2'
                )
                
                # Determine target unit label for clearer metric names
                target_label = target_variable.replace('_', ' ').title()
                if 'throughput' in target_variable.lower():
                    unit_label = ' (kbps)'
                elif 'latency' in target_variable.lower():
                    unit_label = ' (ms)'
                elif 'ber' in target_variable.lower() or 'packet_loss' in target_variable.lower():
                    unit_label = ' (fraction)'
                elif 'signal' in target_variable.lower():
                    unit_label = ' (normalized)'
                else:
                    unit_label = ''

                # Display metrics with explanations
                st.metric("R² Score (Test)", f"{r2:.4f}",
                         help="Explains how well the model predicts. 1.0 is perfect, >0.85 is excellent")
                st.metric("R² Score (Train)", f"{r2_train:.4f}",
                         help="Training score - compare with test score to check overfitting")
                st.metric(f"RMSE{unit_label}", f"{rmse:.2f}",
                         help=f"Average prediction error{unit_label}. Lower is better")
                st.metric(f"MAE{unit_label}", f"{mae:.2f}",
                         help="Mean Absolute Error - average deviation from actual values")
                st.metric("MAPE (%)", f"{mape:.2f}%",
                         help="Mean Absolute Percentage Error - relative error percentage")
                st.metric("CV R² (mean±std)", f"{cv_scores.mean():.4f}±{cv_scores.std():.4f}",
                         help="Cross-validation score shows model stability across different data splits")
                
                # Interpretation
                if r2 > 0.9:
                    st.success(f"✓ Excellent predictive power (R² > 0.9)")
                elif r2 > 0.8:
                    st.success(f"✓ Very good predictive power (R² > 0.8)")
                elif r2 > 0.7:
                    st.info(f"Good predictive power (R² > 0.7)")
                else:
                    st.warning(f"Moderate predictive power (R² = {r2:.4f})")
                
                # Check overfitting
                if r2_train - r2 > 0.1:
                    st.warning(f"⚠ Possible overfitting detected (train-test gap: {r2_train - r2:.4f})")
                else:
                    st.success(f"✓ Model generalizes well (minimal overfitting)")
                
                performance_results.append({
                    'Model': model_name,
                    'R² (Test)': r2,
                    'R² (Train)': r2_train,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE (%)': mape,
                    'CV R²': cv_scores.mean(),
                    'CV Std': cv_scores.std()
                })
                
                # Track best model
                if r2 > best_r2_score:
                    best_r2_score = r2
                    best_model_name = model_name
                    best_model = model
                
                # Store predictions and actuals for all models for later analysis
                if idx == 0:  # First model
                    all_predictions = {}
                    all_actuals = y_test
                all_predictions[model_name] = predictions
                
                # Prediction visualization for Linear Regression
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
                        name='Perfect Prediction Line',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title=f"{model_name}: Predicted vs Actual Values",
                        xaxis_title="Actual Values (Signal Strength)",
                        yaxis_title="Predicted Values (Signal Strength)",
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, width='stretch')
    
    # Performance comparison table with styling
    st.write("#### Model Performance Comparison")
    st.write("""
    **How to interpret these metrics:**
    - **R² (Test)**: Percentage of variance explained by the model (higher is better, max 1.0)
    - **R² (Train)**: Training set performance (compare with test to detect overfitting)
    - **RMSE**: Root Mean Square Error (same units as the target variable)
    - **MAE**: Mean Absolute Error (same units as the target variable)
    - **MAPE**: Mean Absolute Percentage Error (relative accuracy)
    - **CV R²**: Cross-validation R² score (model stability indicator)
    - **CV Std**: Standard deviation of CV scores (lower means more consistent)
    """)
    
    results_df = pd.DataFrame(performance_results)
    
    # Style the dataframe to highlight best values
    def highlight_best(s):
        if s.name in ['R² (Test)', 'R² (Train)', 'CV R²']:
            is_max = s == s.max()
            return ['background-color: lightgreen' if v else '' for v in is_max]
        elif s.name in ['RMSE', 'MAE', 'MAPE (%)', 'CV Std']:
            is_min = s == s.min()
            return ['background-color: lightgreen' if v else '' for v in is_min]
        return ['' for _ in s]
    
    styled_df = results_df.style.apply(highlight_best)
    st.dataframe(styled_df, width='stretch', hide_index=True)
    
    # Detailed performance interpretation
    st.write("#### Performance Interpretation")
    best_result = results_df.loc[results_df['R² (Test)'].idxmax()]

    # Prepare human-friendly labels for the target variable and units
    target_label = target_variable.replace('_', ' ').title()
    if 'throughput' in target_variable.lower():
        unit_label = 'kbps'
    elif 'latency' in target_variable.lower():
        unit_label = 'ms'
    elif 'ber' in target_variable.lower() or 'packet_loss' in target_variable.lower():
        unit_label = 'fraction'
    elif 'signal' in target_variable.lower():
        unit_label = 'normalized'
    else:
        unit_label = ''

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Best Model Analysis:**")
        st.write(f"- Model: **{best_result['Model']}**")
        st.write(f"- Test R² Score: **{best_result['R² (Test)']:.4f}**")
        st.write(f"- This means the model explains **{best_result['R² (Test)']*100:.2f}%** of variance in {target_label}")
        st.write(f"- Average prediction error: **±{best_result['MAE']:.2f} {unit_label}**")
        st.write(f"- Relative error: **{best_result['MAPE (%)']:.2f}%**")
    
    with col2:
        st.write("**What this means for your network:**")
        if best_result['R² (Test)'] > 0.9:
            st.success("✓ **Excellent predictive accuracy** - Model is highly reliable for network planning")
        elif best_result['R² (Test)'] > 0.8:
            st.success("✓ **Very good accuracy** - Suitable for production use")
        elif best_result['R² (Test)'] > 0.7:
            st.info("✓ **Good accuracy** - Reliable for most scenarios")
        else:
            st.warning("⚠ **Moderate accuracy** - Use with caution in critical decisions")
        
        if best_result['R² (Train)'] - best_result['R² (Test)'] < 0.05:
            st.success("✓ **No overfitting detected** - Model generalizes well to new data")
        else:
            st.warning(f"⚠ **Some overfitting** - Train-test gap: {best_result['R² (Train)'] - best_result['R² (Test)']:.4f}")
        
        if best_result['CV Std'] < 0.05:
            st.success("✓ **Highly stable** - Consistent performance across data splits")
        else:
            st.info(f"Stability: CV std = {best_result['CV Std']:.4f}")
    
    # Residual Analysis for Best Model
    st.write(f"### 📊 Residual Analysis - {best_model_name}")
    best_predictions = all_predictions[best_model_name]
    residuals = all_actuals - best_predictions
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residual Histogram
        fig_residuals = go.Figure()
        fig_residuals.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Residuals',
            marker=dict(color='steelblue', line=dict(color='darkblue', width=1))
        ))
        fig_residuals.update_layout(
            title="Prediction Error Distribution (Residuals)",
            xaxis_title=f"Residual Error ({unit_label})",
            yaxis_title="Frequency",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_residuals, width='stretch')
        
        # Calculate residual statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        st.write("**Residual Statistics:**")
        st.write(f"- Mean: {mean_residual:.4f} {unit_label} (bias)")
        st.write(f"- Std Dev: {std_residual:.4f} {unit_label}")
        
        if abs(mean_residual) < std_residual * 0.1:
            st.success("✓ **Low bias** - Errors centered around zero")
        else:
            st.warning("⚠ **Some bias detected** - Model tends to over/underpredict")
    
    with col2:
        # Calibration Plot
        fig_calibration = go.Figure()
        fig_calibration.add_trace(go.Scatter(
            x=all_actuals,
            y=best_predictions,
            mode='markers',
            name='Predictions',
            marker=dict(size=6, opacity=0.5, color='steelblue')
        ))
        
        # Perfect prediction line
        min_val = min(all_actuals.min(), best_predictions.min())
        max_val = max(all_actuals.max(), best_predictions.max())
        fig_calibration.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig_calibration.update_layout(
            title="Model Calibration (Predicted vs Actual)",
            xaxis_title=f"Actual {target_label}",
            yaxis_title=f"Predicted {target_label}",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_calibration, width='stretch')
        
        st.write("**Calibration Quality:**")
        st.write("Points close to the red line indicate well-calibrated predictions")
        if best_result['R² (Test)'] > 0.85:
            st.success("✓ **Excellent calibration** - Predictions align well with actuals")
        elif best_result['R² (Test)'] > 0.7:
            st.info("ℹ **Good calibration** - Generally reliable predictions")
        else:
            st.warning("⚠ **Moderate calibration** - Some prediction drift observed")
    
    # Highlight best model with detailed explanation
    st.success(
        f"🏆 Best Performing Model: **{best_model_name}** | "
        f"R² Score: {best_r2_score:.4f} | "
        f"Prediction Accuracy: {(1 - best_result['MAPE (%)']/100)*100:.2f}%"
    )
    
    # Feature importance visualization
    if hasattr(best_model, 'feature_importances_'):
        st.write(f"#### Feature Importance Analysis - {best_model_name}")
        plot_feature_importance(
            X.columns,
            best_model.feature_importances_,
            best_model_name
        )

    # Save best model
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(best_model, file)
    st.info(
        f"Model '{best_model_name}' persisted to {MODEL_FILE} "
        "for production deployment"
    )
    
    return best_model


def plot_feature_importance(features, importances, model_name):
    """
    Visualize feature importance from tree-based models.
    
    Parameters:
    -----------
    features : array-like
        Feature names
    importances : array-like
        Feature importance scores
    model_name : str
        Name of the model
    """
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance Score': importances
    }).sort_values('Importance Score', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=importance_df['Importance Score'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=importance_df['Importance Score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        )
    ))
    
    fig.update_layout(
        title=f"{model_name} - Feature Importance Rankings",
        xaxis_title="Importance Score",
        yaxis_title="Network Parameters",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, width='stretch')


def visualize_network_performance(data):
    """
    Create advanced 3D and 2D visualizations of network performance.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Network telemetry data
    """
    st.subheader("Network Performance Visualization")
    st.write("""
    **Visual Analysis Purpose**: These visualizations reveal the relationships between 
    network parameters and help identify optimal operating conditions.
    
    **Key Insights to Look For:**
    - How SNR affects throughput and error rates
    - Performance differences between modulation schemes
    - Impact of distance on signal quality
    - Trade-offs between latency and throughput
    """)
    
    # Determine column names
    snr_col = 'snr_db' if 'snr_db' in data.columns else 'snr'
    throughput_col = ('throughput_kbps' if 'throughput_kbps' 
                     in data.columns else 'throughput')
    latency_col = 'latency_ms' if 'latency_ms' in data.columns else 'latency'
    mod_col = ('modulation_scheme' if 'modulation_scheme' 
              in data.columns else 'modulation')
    node_col = 'node_id' if 'node_id' in data.columns else 'node'
    dist_col = ('gateway_distance_km' if 'gateway_distance_km' 
               in data.columns else 'distance_to_gateway')
    packet_loss_col = ('packet_loss_rate' if 'packet_loss_rate' 
                      in data.columns else 'packet_loss')
    sf_col = 'spreading_factor'
    
    # 3D scatter plot with detailed explanation
    st.write("#### 3D Performance Space Analysis")
    st.write("""
    **What this shows**: The 3D plot visualizes how three critical metrics interact:
    - **X-axis (SNR)**: Signal quality in decibels - higher is better
    - **Y-axis (Throughput)**: Data transmission rate in kbps - higher is better  
    - **Z-axis (BER)**: Bit Error Rate - lower is better
    - **Colors**: Different modulation schemes (CSS, FSK, LoRa)
    - **Size**: Represents latency - larger bubbles indicate higher latency
    - **Animation**: Shows changes across different spreading factors
    
    **How to interpret**: Look for clusters where high SNR corresponds with 
    high throughput and low BER - these represent optimal operating conditions.
    """)
    
    fig_3d = px.scatter_3d(
        data,
        x=snr_col,
        y=throughput_col,
        z='ber',
        color=mod_col,
        size=latency_col,
        animation_frame=sf_col if sf_col in data.columns else None,
        hover_data=[node_col, dist_col, packet_loss_col],
        title='3D Network Performance Space: SNR vs Throughput vs BER',
        labels={
            snr_col: 'Signal-to-Noise Ratio (dB)',
            throughput_col: 'Network Throughput (kbps)',
            'ber': 'Bit Error Rate',
            mod_col: 'Modulation Scheme',
            latency_col: 'Latency (ms)',
            sf_col: 'Spreading Factor (SF)'
        },
        opacity=0.7,
        color_discrete_map={
            'CSS': '#1f77b4',
            'FSK': '#ff7f0e',
            'LoRa': '#2ca02c'
        }
    )
    
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='Signal-to-Noise Ratio (dB)',
            yaxis_title='Network Throughput (kbps)',
            zaxis_title='Bit Error Rate',
            xaxis=dict(backgroundcolor="rgb(230, 230, 230)"),
            yaxis=dict(backgroundcolor="rgb(230, 230, 230)"),
            zaxis=dict(backgroundcolor="rgb(230, 230, 230)")
        ),
        height=700,
        font=dict(size=11)
    )
    
    st.plotly_chart(fig_3d, width='stretch')
    
    # 2D analysis plots
    st.write("#### Two-Dimensional Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SNR vs Distance analysis
        st.write("""
        **Distance vs Signal Quality:**
        Shows how signal-to-noise ratio decreases with distance from gateway.
        LOWESS trendlines reveal the smoothed signal decay pattern for each scheme.
        Look for steeper slopes - they indicate faster signal degradation.
        """)
        fig_snr_dist = px.scatter(
            data,
            x=dist_col,
            y=snr_col,
            color=mod_col,
            size='signal_strength',
            trendline='lowess',
            title='Signal Quality vs Distance: Path Loss Analysis',
            labels={
                dist_col: 'Distance to Gateway (km)',
                snr_col: 'Signal-to-Noise Ratio (dB)',
                mod_col: 'Modulation Scheme',
                'signal_strength': 'Signal Strength'
            },
            hover_data=[node_col, throughput_col]
        )
        fig_snr_dist.update_layout(height=400)
        st.plotly_chart(fig_snr_dist, width='stretch')
    
    with col2:
        # Throughput vs Latency trade-off
        st.write("""
        **Throughput vs Latency Trade-off:**
        Ideal points are in the upper-left (high throughput, low latency).
        Bubble size shows packet loss - smaller bubbles mean better reliability.
        LOWESS trendlines highlight the overall performance trade-off pattern.
        """)
        fig_tp_lat = px.scatter(
            data,
            x=latency_col,
            y=throughput_col,
            color=mod_col,
            size=packet_loss_col,
            trendline='lowess',
            title='Performance Trade-off: Throughput vs Latency',
            labels={
                latency_col: 'Network Latency (ms)',
                throughput_col: 'Throughput (kbps)',
                mod_col: 'Modulation Scheme',
                packet_loss_col: 'Packet Loss Rate'
            },
            hover_data=[node_col, snr_col]
        )
        fig_tp_lat.update_layout(height=400)
        st.plotly_chart(fig_tp_lat, width='stretch')
    
    # Time series if timestamp available
    # Ensure timestamp column is a pandas datetime to avoid serialization issues
    data = data.copy()
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        st.write("#### Real-Time Network Monitoring")
        fig_ts = go.Figure()

        for node in data[node_col].unique():
            node_data = data[data[node_col] == node].dropna(subset=['timestamp'])
            if node_data.empty:
                continue
            fig_ts.add_trace(go.Scatter(
                x=node_data['timestamp'],
                y=node_data[throughput_col],
                mode='lines+markers',
                name=node,
                line=dict(width=2)
            ))

        fig_ts.update_layout(
            title='Time-Series Analysis: Real-Time Throughput Monitoring',
            xaxis_title='Timestamp',
            yaxis_title='Throughput (kbps)',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_ts, width='stretch')

        # Real-time snapshot KPIs (last windows)
        recent = data.sort_values('timestamp')
        last_window = recent.tail(30)
        prev_window = recent.tail(60).head(30)
        if len(last_window) >= 5:
            current_mean = last_window[throughput_col].mean()
            prev_mean = prev_window[throughput_col].mean() if len(prev_window) > 0 else None
            delta_text = None
            if prev_mean is not None:
                delta_val = current_mean - prev_mean
                delta_text = f"{delta_val:.2f} kbps"

            k1, k2 = st.columns(2)
            with k1:
                st.metric("Current Avg Throughput (last 30 samples)", f"{current_mean:.2f} kbps", delta=delta_text)
            with k2:
                st.write("_Note: Throughput snapshot uses the most recent time window to indicate short-term changes._")


def analyze_modulation_schemes(data):
    """
    Comprehensive analysis of modulation scheme performance.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Network telemetry data
        
    Returns:
    --------
    str
        Recommended modulation scheme
    """
    st.subheader("Modulation Scheme Performance Analysis")
    st.write("""
    **Analysis Purpose**: Compare different modulation schemes to identify which 
    performs best under current network conditions.
    
    **Modulation Schemes Analyzed:**
    - **CSS (Chirp Spread Spectrum)**: Good for long-range, robust against interference
    - **FSK (Frequency Shift Keying)**: Balance of range and data rate
    - **LoRa (Long Range)**: Optimized for low power, long-range IoT applications
    
    **Key Performance Indicators:**
    - Throughput: Higher values indicate better data transmission capacity
    - Latency: Lower values mean faster response times
    - BER: Lower bit error rates indicate more reliable communication
    - Packet Loss: Lower loss rates ensure better data integrity
    """)
    
    # Determine column names
    snr_col = 'snr_db' if 'snr_db' in data.columns else 'snr'
    throughput_col = ('throughput_kbps' if 'throughput_kbps' 
                     in data.columns else 'throughput')
    latency_col = 'latency_ms' if 'latency_ms' in data.columns else 'latency'
    mod_col = ('modulation_scheme' if 'modulation_scheme' 
              in data.columns else 'modulation')
    dist_col = ('gateway_distance_km' if 'gateway_distance_km' 
               in data.columns else 'distance_to_gateway')
    packet_loss_col = ('packet_loss_rate' if 'packet_loss_rate' 
                      in data.columns else 'packet_loss')
    bw_col = 'bandwidth_khz' if 'bandwidth_khz' in data.columns else 'bandwidth'
    
    summary = data.groupby(mod_col).agg({
        snr_col: 'mean',
        'ber': 'mean',
        throughput_col: 'mean',
        'data_rate': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A',
        'spreading_factor': 'mean',
        bw_col: 'mean',
        latency_col: 'mean',
        packet_loss_col: 'mean',
        dist_col: 'mean'
    }).reset_index()
    
    summary['modulation_full_name'] = summary[mod_col].map(
        MODULATION_SCHEMES
    )
    
    # Display comprehensive metrics
    st.write("#### Comparative Performance Metrics by Modulation Scheme")
    summary_display = summary.copy()
    summary_display.columns = [
        "Modulation", "Avg SNR (dB)", "Avg BER",
        "Avg Throughput (kbps)", "Primary Data Rate",
        "Avg SF", "Avg Bandwidth (kHz)", "Avg Latency (ms)",
        "Avg Packet Loss", "Avg Distance (km)", "Full Name"
    ]
    
    numeric_cols = summary_display.select_dtypes(include=[np.number]).columns
    summary_display[numeric_cols] = summary_display[numeric_cols].round(3)
    
    st.dataframe(summary_display, width='stretch', hide_index=True)

    # Comparative visualizations
    st.write("#### Performance Comparison Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_throughput = px.bar(
            summary,
            x='modulation_full_name',
            y=throughput_col,
            color='modulation_full_name',
            title='Average Throughput by Modulation Scheme',
            labels={
                'modulation_full_name': 'Modulation Type',
                throughput_col: 'Average Throughput (kbps)'
            },
            text_auto='.2f'
        )
        fig_throughput.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_throughput, width='stretch')
    
    with col2:
        fig_latency = px.bar(
            summary,
            x='modulation_full_name',
            y=latency_col,
            color='modulation_full_name',
            title='Average Latency by Modulation Scheme',
            labels={
                'modulation_full_name': 'Modulation Type',
                latency_col: 'Average Latency (ms)'
            },
            text_auto='.2f'
        )
        fig_latency.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_latency, width='stretch')

    # Identify optimal modulation
    best_modulation = summary.loc[summary[throughput_col].idxmax()]
    
    st.write("#### Optimal Modulation Recommendation")
    
    # Calculate performance score
    throughput_norm = (
        (best_modulation[throughput_col] - summary[throughput_col].min()) /
        (summary[throughput_col].max() - summary[throughput_col].min())
    )
    latency_norm = 1 - (
        (best_modulation[latency_col] - summary[latency_col].min()) /
        (summary[latency_col].max() - summary[latency_col].min())
    )
    performance_score = (throughput_norm * 0.6 + latency_norm * 0.4) * 100
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(
            f"""
            <div style="background-color: #e8f5e9; padding: 20px;
                        border-radius: 10px; border-left: 5px solid #4caf50;">
                <h4 style="color: #2e7d32; margin-top: 0;">
                    Recommended: {best_modulation['modulation_full_name']}
                </h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li><strong>Average SNR:</strong> {best_modulation[snr_col]:.2f} dB</li>
                    <li><strong>Bit Error Rate:</strong> {best_modulation['ber']:.4f}</li>
                    <li><strong>Throughput:</strong> {best_modulation[throughput_col]:.2f} kbps</li>
                    <li><strong>Latency:</strong> {best_modulation[latency_col]:.2f} ms</li>
                    <li><strong>Packet Loss:</strong> {best_modulation[packet_loss_col]:.4f}</li>
                    <li><strong>Coverage Range:</strong> {best_modulation[dist_col]:.2f} km</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.metric(
            "Performance Score",
            f"{performance_score:.1f}%",
            delta="Optimal"
        )
    
    with col3:
        efficiency = (
            best_modulation[throughput_col] / best_modulation[latency_col]
        )
        st.metric(
            "Efficiency Rating",
            f"{efficiency:.2f}",
            delta="kbps/ms"
        )
    
    # Detailed interpretation
    st.write("#### What This Recommendation Means")
    st.write(f"""
    **Selected Modulation: {best_modulation['modulation_full_name']}**
    
    **Performance Analysis:**
    - This modulation scheme achieved the highest throughput of **{best_modulation[throughput_col]:.2f} kbps**
    - Average latency of **{best_modulation[latency_col]:.2f} ms** provides {
        'excellent' if best_modulation[latency_col] < 40 else 
        'good' if best_modulation[latency_col] < 60 else 'acceptable'
    } response time
    - Bit Error Rate of **{best_modulation['ber']:.4f}** indicates {
        'excellent' if best_modulation['ber'] < 0.03 else 
        'good' if best_modulation['ber'] < 0.05 else 'acceptable'
    } signal quality
    - Packet loss rate of **{best_modulation[packet_loss_col]:.2%}** ensures {
        'high' if best_modulation[packet_loss_col] < 0.03 else 
        'good' if best_modulation[packet_loss_col] < 0.05 else 'acceptable'
    } data reliability
    
    **Recommended Use Cases:**
    """)
    
    # Use case recommendations based on modulation type
    if best_modulation[mod_col] == 'LoRa':
        st.info("""
        **LoRa is optimal for:**
        - Long-range IoT sensor networks
        - Low-power battery-operated devices
        - Rural or remote area coverage
        - Applications requiring extended battery life
        """)
    elif best_modulation[mod_col] == 'FSK':
        st.info("""
        **FSK is optimal for:**
        - Balanced range and data rate requirements
        - Industrial monitoring applications
        - Moderate data throughput scenarios
        - Cost-effective implementations
        """)
    elif best_modulation[mod_col] == 'CSS':
        st.info("""
        **CSS is optimal for:**
        - High interference environments
        - Robust long-range communications
        - Applications requiring resilience
        - Urban deployments with noise
        """)
    
    st.write(f"""
    **Overall Performance Score: {performance_score:.1f}/100**
    - Throughput contribution: {throughput_norm * 60:.1f}/60 points
    - Latency contribution: {latency_norm * 40:.1f}/40 points
    - Efficiency ratio: {efficiency:.2f} kbps per millisecond of latency
    
    **Deployment Recommendation:** This modulation scheme is recommended for 
    production deployment based on superior throughput performance while maintaining 
    acceptable latency and error rates.
    """)
    
    return best_modulation[mod_col]


def analyze_network_logs(log_entries):
    """
    Analyze network logs for pattern detection and anomalies.
    
    Parameters:
    -----------
    log_entries : list
        Network log entries
    """
    st.subheader("Network Log Analysis")
    
    cleaned_logs = [re.sub(r'\W+', ' ', entry) for entry in log_entries]
    vectorizer = CountVectorizer(max_features=10)
    log_features = vectorizer.fit_transform(cleaned_logs)
    
    st.write("#### Recent Log Entries")
    for idx, log in enumerate(log_entries, 1):
        is_critical = ("critical" in log.lower() or "low" in log.lower() 
                      or "high" in log.lower() or "timeout" in log.lower())
        log_type = "[WARNING]" if is_critical else "[INFO]"
        log_color = "#ff6b6b" if is_critical else "#4ecdc4"
        
        st.markdown(
            f"<div style='background-color: {log_color}; padding: 10px; "
            f"border-radius: 5px; margin: 5px 0; color: white;'>"
            f"<strong>{log_type}</strong> Entry {idx}: {log}</div>",
            unsafe_allow_html=True
        )
    
    st.write("#### Extracted Keywords and Patterns")
    feature_names = vectorizer.get_feature_names_out()
    st.write(", ".join(feature_names))
    
    st.info(
        "Log analysis complete. Pattern extraction performed for "
        "anomaly detection and network diagnostics."
    )


def calculate_network_efficiency(data):
    """
    Calculate and visualize network efficiency metrics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Network telemetry data
        
    Returns:
    --------
    float
        Average network efficiency
    """
    st.subheader("Network Efficiency Analysis")
    
    # Determine column names
    throughput_col = ('throughput_kbps' if 'throughput_kbps' 
                     in data.columns else 'throughput')
    latency_col = 'latency_ms' if 'latency_ms' in data.columns else 'latency'
    
    data['efficiency'] = data[throughput_col] / data[latency_col]
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
        improvement_pct = ((max_efficiency / avg_efficiency - 1) * 100)
        st.metric(
            "Peak Efficiency",
            f"{max_efficiency:.3f}",
            delta=f"+{improvement_pct:.1f}%"
        )
    
    with col3:
        utilization = (data[throughput_col].mean() / 150) * 100
        st.metric(
            "Network Utilization",
            f"{utilization:.1f}%",
            delta="of theoretical capacity"
        )
    
    # Efficiency distribution
    fig_efficiency = px.histogram(
        data,
        x='efficiency',
        nbins=50,
        title='Network Efficiency Distribution Analysis',
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
    Main application entry point for LoRa network analysis platform.
    """
    st.title("LoRa Network Analysis & Optimization Platform")
    
    st.markdown("""
        <div style="background-color: #e3f2fd; padding: 20px;
                    border-radius: 10px; margin-bottom: 20px;
                    border-left: 5px solid #2196f3;">
            <h4 style="color: #1565c0; margin-top: 0;">
                Advanced Network Analytics for LoRaWAN Deployments
            </h4>
            <p style="color: #424242; margin-bottom: 0;">
                Comprehensive platform for LoRa network monitoring, predictive
                analytics, and performance optimization. Leverage machine learning
                and statistical analysis for data-driven network management decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Configuration sidebar
    with st.sidebar:
        st.header("Configuration Panel")
        
        st.write("### Data Generation Settings")
        real_time_mode = st.checkbox(
            "Enable Real-Time Simulation",
            value=False,
            help="Generate time-stamped data for time-series analysis"
        )
        
        sample_size = st.slider(
            "Sample Size",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Number of telemetry data points to generate"
        )
        
        st.write("### Data Preprocessing")
        imputation_strategy = st.selectbox(
            "Missing Value Imputation",
            ["mean", "median"],
            help="Strategy for handling missing values"
        )
        
        scaling_method = st.selectbox(
            "Feature Scaling Method",
            ["standard", "minmax"],
            help="Normalization technique for feature scaling"
        )
        
        if real_time_mode:
            st.write("#### 🔴 Live Streaming Demo Mode")
            auto_refresh = st.checkbox(
                "Enable Continuous Data Streaming",
                value=False,
                help="Continuously generates and updates with new data points"
            )
            
            if auto_refresh:
                refresh_interval = st.slider(
                    "Streaming Interval (seconds)",
                    min_value=5,
                    max_value=60,
                    value=10,
                    step=5,
                    help="How often to generate and append new data"
                )
                
                # Initialize session state for streaming data
                if 'streaming_data' not in st.session_state:
                    st.session_state.streaming_data = None
                    st.session_state.stream_count = 0
                
                time.sleep(refresh_interval)
                st.session_state.stream_count += 1
                st.rerun()
    
    # Generate network data
    with st.spinner('Generating network telemetry data...'):
        # In streaming mode, accumulate data over time
        if real_time_mode and auto_refresh and st.session_state.streaming_data is not None:
            # Generate new batch and append
            new_batch = generate_network_data(
                samples=min(50, sample_size // 10),  # Smaller batch
                real_time_mode=real_time_mode
            )
            network_data = pd.concat(
                [st.session_state.streaming_data, new_batch],
                ignore_index=True
            ).tail(sample_size)  # Keep only most recent samples
            st.session_state.streaming_data = network_data
            
            st.info(f"🔴 Live Stream #{st.session_state.stream_count}: Added {len(new_batch)} new samples")
        else:
            # Normal generation or first streaming initialization
            network_data = generate_network_data(
                samples=sample_size,
                real_time_mode=real_time_mode
            )
            if real_time_mode and auto_refresh:
                st.session_state.streaming_data = network_data
    
    st.success(
        f"Successfully generated {len(network_data)} telemetry samples"
    )
    
    # Data overview
    with st.expander("View Raw Data Summary", expanded=False):
        st.write("#### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        node_col = 'node_id' if 'node_id' in network_data.columns else 'node'
        mod_col = ('modulation_scheme' if 'modulation_scheme' 
                  in network_data.columns else 'modulation')
        
        with col1:
            st.metric("Total Samples", len(network_data))
        with col2:
            st.metric("Active Nodes", network_data[node_col].nunique())
        with col3:
            st.metric("Modulation Types", network_data[mod_col].nunique())
        with col4:
            st.metric("Features", len(network_data.columns))
        
        display_df = network_data.head(10).copy()
        if 'timestamp' in display_df.columns:
            # Convert timestamps to string to avoid pyarrow serialization warnings
            display_df['timestamp'] = display_df['timestamp'].astype(str)
        st.dataframe(display_df, width='stretch')
        st.write(network_data.describe())
    
    # Network visualization
    st.header("Network Performance Visualization")
    visualize_network_performance(network_data)

    # Preprocess data
    with st.spinner('Preprocessing telemetry data...'):
        processed_data = preprocess_telemetry_data(
            network_data.copy(),
            imputation_strategy=imputation_strategy,
            scaling_method=scaling_method
        )
    
    # Exploratory analysis
    st.header("Exploratory Data Analysis")
    perform_exploratory_analysis(processed_data)

    # Machine learning
    st.header("Predictive Model Training")
    
    with st.expander("About Predictive Models", expanded=False):
        st.write("""
            Train regression models to predict network signal strength based on
            various parameters. Models are evaluated using:
            
            - **R² Score**: Coefficient of determination (0-1, higher is better)
            - **RMSE**: Root Mean Squared Error (lower is better)
            - **MAE**: Mean Absolute Error (lower is better)
            - **CV Score**: 5-fold cross-validation score for robustness
        """)
    
    if st.button("Train Predictive Models", type="primary"):
        with st.spinner('Training machine learning models...'):
            # Attach original (unscaled) target column to the preprocessed features
            processed_for_training = processed_data.copy()
            target_col = 'signal_strength'
            if target_col in network_data.columns:
                # Use original units for the target so metrics are reported in realistic units
                processed_for_training[target_col] = network_data[target_col].values

            best_model = train_predictive_models(
                processed_for_training,
                target_col
            )
    
    # Modulation analysis
    st.header("Modulation Scheme Analysis")
    
    with st.expander("About Modulation Analysis", expanded=False):
        st.write("""
            Comparative analysis of LoRa modulation schemes:
            
            - **CSS (Chirp Spread Spectrum)**: Long range, robust
            - **FSK (Frequency-Shift Keying)**: Simple, reliable
            - **LoRa**: Proprietary, optimized for LPWAN
            
            Analysis considers throughput, latency, error rates, and coverage.
        """)
    
    if st.button("Analyze Modulation Schemes", type="primary"):
        with st.spinner('Analyzing modulation performance...'):
            recommended_modulation = analyze_modulation_schemes(network_data)
    
    # Network logs
    st.header("Network Diagnostics & Log Analysis")
    
    sample_logs = [
        "Node1: Signal degradation detected - Distance: 15.2 km, SNR: 8.3 dB",
        "Node2: Battery level critical - Immediate maintenance required",
        "Node3: High packet loss rate - BER exceeds threshold: 0.085",
        "Node4: Optimal performance - All parameters within normal range",
        "Gateway: Connection timeout detected on Node2 - Retry attempt initiated"
    ]
    
    if st.button("Analyze Network Logs"):
        analyze_network_logs(sample_logs)
    
    # Efficiency metrics
    st.header("Network Efficiency Metrics")
    efficiency = calculate_network_efficiency(network_data)
    
    # Summary with comprehensive result interpretation
    st.header("Analysis Summary & Key Insights")
    
    st.write("""
    ### Understanding Your Network Analysis Results
    
    This comprehensive analysis provides actionable insights for optimizing your 
    LoRa network performance. Here's what each component reveals:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 Data Analysis Results
        
        **Network Telemetry Statistics:**
        - Generated **{:,} samples** covering various network conditions
        - **SNR Range**: Indicates signal quality variations across coverage area
        - **Throughput Distribution**: Shows typical data transmission rates
        - **Latency Patterns**: Reveals response time characteristics
        
        **What this means:** The statistical distributions show your network's 
        operational envelope and identify potential bottlenecks.
        
        #### 🤖 Machine Learning Predictions
        
        **Model Performance Metrics Explained:**
        - **R² Score (>0.85)**: Model explains 85%+ of throughput variations
        - **RMSE (kbps)**: Average prediction error magnitude
        - **MAE (kbps)**: Typical deviation from actual values
        - **Cross-Validation**: Confirms model reliability
        
        **Practical Application:** Use these models to predict network performance 
        before deploying new nodes or changing configurations.
        
        #### 📡 Modulation Optimization
        
        **Recommendation Basis:**
        - Throughput maximization (60% weight)
        - Latency minimization (40% weight)
        - Error rate considerations
        - Coverage area assessment
        
        **Implementation:** The recommended modulation scheme offers the best 
        balance between speed, reliability, and range for your deployment scenario.
        """.format(len(network_data)))
    
    with col2:
        st.markdown("""
        #### 🎯 Performance Benchmarks
        
        **Excellent Performance Indicators:**
        - ✓ R² Score > 0.9 (Outstanding prediction accuracy)
        - ✓ MAPE < 5% (High precision)
        - ✓ BER < 0.03 (Reliable communication)
        - ✓ Packet Loss < 3% (Strong data integrity)
        - ✓ SNR > 15 dB (Excellent signal quality)
        
        **Good Performance Indicators:**
        - ✓ R² Score 0.8-0.9 (Very good predictions)
        - ✓ MAPE 5-10% (Good precision)
        - ✓ BER 0.03-0.05 (Good reliability)
        - ✓ Packet Loss 3-5% (Acceptable)
        - ✓ SNR 10-15 dB (Good signal)
        
        #### 💡 Actionable Recommendations
        
        **Based on Analysis Results:**
        
        1. **If High R² Score (>0.9):**
           - Deploy models for predictive maintenance
           - Use for capacity planning
           - Implement automated optimization
        
        2. **If Low BER (<0.03):**
           - Current modulation scheme is optimal
           - Consider expanding coverage area
           - Network is production-ready
        
        3. **If High Throughput (>100 kbps):**
           - Network can support data-intensive applications
           - Consider adding more nodes
           - Explore real-time monitoring capabilities
        
        4. **Efficiency Optimization:**
           - Monitor nodes with low efficiency scores
           - Adjust spreading factors for distant nodes
           - Balance power consumption with performance
        """)
    
    st.success("""
    ### ✅ Next Steps for Network Optimization
    
    **Immediate Actions:**
    1. Review the model comparison table - use the highest R² model for predictions
    2. Check the modulation recommendation - implement suggested scheme
    3. Analyze the 3D visualization - identify performance clusters
    4. Monitor nodes with efficiency scores below 0.7
    
    **Long-term Strategy:**
    1. Collect real network data to retrain models
    2. Establish performance baselines using current metrics
    3. Implement continuous monitoring with automated alerts
    4. Plan network expansion based on coverage analysis
    5. Optimize spreading factors for each deployment zone
    
    **Performance Monitoring:**
    - Set SNR threshold alerts at < 10 dB
    - Monitor packet loss rates (alert if > 5%)
    - Track throughput degradation patterns
    - Review modulation efficiency quarterly
    """)
    
    
    # Footer with results interpretation guide
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p><strong>LoRa Network Analysis & Optimization Platform</strong></p>
            <p style="font-size: 0.9em;">
                Advanced Analytics | Machine Learning | Real-Time Monitoring
            </p>
            <p style="font-size: 0.85em;">
                <strong>Result Interpretation Guide:</strong><br>
                R² > 0.9 = Excellent | 0.8-0.9 = Very Good | 0.7-0.8 = Good<br>
                SNR > 15dB = Excellent | 10-15dB = Good | <10dB = Needs Attention<br>
                Packet Loss < 3% = Excellent | 3-5% = Good | >5% = Investigate
            </p>
            <p style="font-size: 0.85em;">
                Last Updated: {} | Samples Analyzed: {} | Status: Operational
            </p>
        </div>
    """.format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        len(network_data)
    ), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
