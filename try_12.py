import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import math

# Configuration
MODEL_FILE = 'lora_interference_model.pkl'
MAX_DISTANCE = 30  # kilometers
CARRIER_FREQ = 868e6  # 868 MHz for EU LoRa

MODULATION_PARAMS = {
    'SF7': {'sensitivity': -123, 'data_rate': 5.48},
    'SF8': {'sensitivity': -126, 'data_rate': 3.13},
    'SF9': {'sensitivity': -129, 'data_rate': 1.76},
    'SF10': {'sensitivity': -132, 'data_rate': 0.98},
    'SF11': {'sensitivity': -134, 'data_rate': 0.44},
    'SF12': {'sensitivity': -136, 'data_rate': 0.23}
}

def path_loss(distance, frequency=CARRIER_FREQ):
    """Calculate free space path loss"""
    return 20 * math.log10(distance) + 20 * math.log10(frequency) + 20 * math.log10(4 * math.pi / 3e8)

def create_synthetic_data(samples=1000):
    np.random.seed(42)
    data = {
        'distance': np.random.uniform(0.1, MAX_DISTANCE, samples),
        'tx_power': np.random.choice([14, 17, 20], samples),  # dBm
        'sf': np.random.choice(list(MODULATION_PARAMS.keys()), samples),
        'interference': np.random.uniform(0, 1, samples),  # Normalized interference level
        'temperature': np.random.uniform(-20, 40, samples),
        'humidity': np.random.uniform(10, 90, samples),
    }
    
    # Calculate derived metrics
    df = pd.DataFrame(data)
    df['path_loss'] = df['distance'].apply(path_loss)
    df['rssi'] = df['tx_power'] - df['path_loss'] + np.random.normal(0, 2, samples)
    df['snr'] = np.clip(-15 * df['distance']/MAX_DISTANCE + 20 * (1 - df['interference']) + np.random.normal(0, 2, samples), -20, 20)
    df['ber'] = np.clip(1e-4 * np.exp(df['distance']/5 + df['interference']*3) + np.random.normal(0, 1e-5, samples), 1e-6, 1)
    df['throughput'] = df['sf'].apply(lambda x: MODULATION_PARAMS[x]['data_rate']) * (1 - df['ber']) * (1 - df['interference'])
    df['success_rate'] = np.clip(0.95 - 0.3*df['distance']/MAX_DISTANCE - 0.6*df['interference'] + np.random.normal(0, 0.05, samples), 0, 1)
    
    # Create interference categories
    df['interference_level'] = pd.cut(df['interference'], 
                                    bins=[0, 0.3, 0.7, 1],
                                    labels=['low', 'medium', 'high'])
    return df

def plot_interference_impact(df):
    st.subheader("Interference Impact Analysis")
    st.markdown("""
    **Key Observations:**
    - Throughput decreases exponentially with interference increase
    - BER shows non-linear relationship with interference
    - Higher interference leads to wider success rate distribution
    """)
    
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=(
                           "Throughput vs Interference",
                           "Bit Error Rate Distribution",
                           "Success Rate by Interference Level",
                           "SNR Distribution"
                       ))
    
    # Throughput vs Interference
    fig.add_trace(go.Scatter(
        x=df['interference'], y=df['throughput'], mode='markers',
        marker=dict(color=df['distance'], colorscale='Viridis', showscale=True,
                   colorbar=dict(title="Distance (km)")),
        name='Throughput'
    ), row=1, col=1)
    
    # BER vs Interference Level
    fig.add_trace(go.Box(
        x=df['interference_level'], y=df['ber'], 
        boxpoints='all', jitter=0.3, name='BER'
    ), row=1, col=2)
    
    # Success Rate Distribution
    fig.add_trace(go.Violin(
        x=df['interference_level'], y=df['success_rate'],
        box_visible=True, meanline_visible=True, name='Success Rate'
    ), row=2, col=1)
    
    # SNR Histogram
    fig.add_trace(go.Histogram(
        x=df['snr'], nbinsx=30, marker_color='#636EFA', name='SNR'
    ), row=2, col=2)
    
    # Update axis labels
    fig.update_xaxes(title_text="Interference Level", row=1, col=1)
    fig.update_yaxes(title_text="Throughput (kbps)", row=1, col=1)
    fig.update_xaxes(title_text="Interference Category", row=1, col=2)
    fig.update_yaxes(title_text="Bit Error Rate", row=1, col=2)
    fig.update_xaxes(title_text="Interference Category", row=2, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", row=2, col=1)
    fig.update_xaxes(title_text="SNR (dB)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    fig.update_layout(
        height=800, 
        title_text="Comprehensive Interference Analysis",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def adaptive_modulation_recommendation(df):
    st.subheader("Adaptive Modulation Strategy")
    st.markdown("""
    **Optimal SF Selection Based on Interference Conditions:**
    - Lower SF for high interference/short range
    - Higher SF for low interference/long range
    """)
    
    best_sf = df.groupby('interference_level').apply(
        lambda x: x.groupby('sf')['success_rate'].mean().idxmax()
    ).reset_index()
    best_sf.columns = ['Interference Level', 'Recommended SF']
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Recommendation Table**")
        st.dataframe(best_sf.style.highlight_max(axis=0))
    
    with col2:
        fig = px.line(
            df.groupby(['sf', 'interference_level'])['success_rate'].mean().reset_index(),
            x='interference_level', y='success_rate', color='sf',
            title="Success Rate by Spreading Factor",
            labels={'interference_level': 'Interference Level', 'success_rate': 'Success Rate (%)'}
        )
        fig.update_layout(
            xaxis_title="Interference Category",
            yaxis_title="Average Success Rate (%)",
            legend_title="Spreading Factor"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    return best_sf

def interference_aware_model(df):
    st.subheader("AI-Powered Throughput Predictor")
    st.markdown("""
    **Random Forest Model for Throughput Prediction:**
    - Trained on network parameters and environmental factors
    - Accounts for interference and spreading factor impacts
    """)
    
    # Prepare data
    # Change this line (add 'r' prefix for raw string):
    df['sf_numeric'] = df['sf'].str.extract(r'(\d+)').astype(int)
    X = df[['distance', 'tx_power', 'interference', 'temperature', 'humidity', 'sf_numeric']]
    y = df['throughput']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    st.write(f"""
    **Model Performance:**
    - R² Score: {r2_score(y_test, preds):.2f}
    - RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f} kbps
    """)
    
    # Feature importance
    fig = px.bar(
        x=X.columns, y=model.feature_importances_,
        labels={'x': 'Feature', 'y': 'Importance'},
        title="Feature Importance Analysis",
        color=model.feature_importances_,
        color_continuous_scale='Blues'
    )
    fig.update_layout(xaxis_title="Network Parameters", yaxis_title="Relative Importance")
    st.plotly_chart(fig, use_container_width=True)
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    
    return model

def main():
    st.set_page_config(page_title="LoRa Network Optimizer", layout="wide")
    st.title("LoRa Network Optimization Dashboard")
    st.markdown("""
    **Comprehensive analysis tool for LoRaWAN network performance optimization**
    """)
    
    # Generate data
    df = create_synthetic_data(1500)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Simulation Controls")
        selected_sf = st.selectbox("Spreading Factor", list(MODULATION_PARAMS.keys()))
        selected_interference = st.slider("Interference Level", 0.0, 1.0, 0.5)
        # In the main() function, modify the slider line:
        selected_distance = st.slider("Distance (km)", 0.1, float(MAX_DISTANCE), 5.0)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Analysis", 
        "📶 Modulation Advisor", 
        "🤖 AI Predictor", 
        "📈 Capacity Insights"
    ])
    
    with tab1:
        plot_interference_impact(df)
        
    with tab2:
        adaptive_modulation_recommendation(df)
        
    with tab3:
        model = interference_aware_model(df)
        
    with tab4:
        st.subheader("Network Capacity Planning")
        fig = px.scatter(
            df.groupby('sf').agg({'distance': 'max', 'throughput': 'mean'}).reset_index(),
            x='distance', y='throughput', size='throughput', color='sf',
            labels={'distance': 'Max Distance (km)', 'throughput': 'Average Throughput (kbps)'},
            title="Throughput vs Coverage Trade-off"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insights:**
        - Higher spreading factors enable longer range but reduce throughput
        - Optimal SF selection depends on application requirements
        """)
    
    # Real-time prediction
    st.sidebar.header("Real-Time Prediction")
    if st.sidebar.button("Predict Current Configuration"):
        input_data = pd.DataFrame([{
            'distance': selected_distance,
            'tx_power': 20,
            'interference': selected_interference,
            'temperature': 25,
            'humidity': 50,
            'sf_numeric': int(selected_sf[2:])
        }])
        
        try:
            prediction = model.predict(input_data)[0]
            st.sidebar.success(f"""
            Predicted Throughput: {prediction:.2f} kbps
            Configuration:
            - SF: {selected_sf}
            - Distance: {selected_distance} km
            - Interference: {selected_interference*100:.0f}%
            """)
        except:
            st.sidebar.warning("Please train the model first using the AI Predictor tab")

if __name__ == "__main__":
    main()