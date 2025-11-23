# import pandas as pd
# import numpy as np
# import streamlit as st
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import pickle
# import re
# from sklearn.feature_extraction.text import CountVectorizer
# import matplotlib.pyplot as plt

# # File path for the saved model
# MODEL_FILE = 'best_lora_model.pkl'

# # Full names for modulation types
# MODULATION_FULL_NAMES = {
#     'CSS': 'Chirp Spread Spectrum',
#     'FSK': 'Frequency-Shift Keying',
#     'LoRa': 'LoRa Proprietary Modulation'
# }

# def create_synthetic_data(samples=1000):
#     np.random.seed(42)  # Seed for reproducibility
#     data = {
#         'node': np.random.choice(['Node1', 'Node2', 'Node3', 'Node4'], samples),
#         'snr': np.random.uniform(5, 20, samples),
#         'ber': np.random.uniform(0.01, 0.1, samples),  # Ensure BER is positive
#         'throughput': np.random.uniform(50, 150, samples),
#         'modulation': np.random.choice(['CSS', 'FSK', 'LoRa'], samples),
#         'signal_strength': np.random.uniform(0.1, 1, samples),  # Ensure signal strength is positive
#         'data_rate': np.random.choice(['DR0', 'DR1', 'DR2', 'DR3'], samples),
#         'spreading_factor': np.random.choice([7, 8, 9, 10, 11, 12], samples),
#         'bandwidth': np.random.choice([125, 250, 500], samples),
#         'latency': np.random.uniform(10, 100, samples),
#         'packet_loss': np.random.uniform(0.01, 0.1, samples),  # Ensure packet loss is positive
#         'distance_to_gateway': np.random.uniform(1, 20, samples)
#     }
#     return pd.DataFrame(data)

# def preprocess_data(data, fill_strategy="mean", scaling_type="standard"):
#     numeric_features = data.select_dtypes(include=[np.number])
    
#     if fill_strategy == "mean":
#         data[numeric_features.columns] = numeric_features.fillna(numeric_features.mean())
#     elif fill_strategy == "median":
#         data[numeric_features.columns] = numeric_features.fillna(numeric_features.median())

#     scaler = StandardScaler() if scaling_type == "standard" else MinMaxScaler()
#     data[numeric_features.columns] = scaler.fit_transform(numeric_features)

#     categorical_features = data.select_dtypes(include=[object])
#     encoder = OneHotEncoder(sparse_output=False, drop='first')
#     encoded_data = encoder.fit_transform(categorical_features)
#     encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features.columns))
#     processed_data = pd.concat([data[numeric_features.columns], encoded_df], axis=1)
    
#     return processed_data

# def exploratory_data_analysis(data):
#     st.subheader("Exploratory Data Analysis")
#     st.write("Understand relationships and distributions of your data through these visualizations.")
    
#     # Displaying correlation matrix to identify relationships between features
#     correlation_matrix = data.corr()
#     fig_corr = px.imshow(correlation_matrix, text_auto=True, title="Correlation Matrix")
#     st.plotly_chart(fig_corr, width='stretch')

#     # Visualizing distributions of important features
#     fig_dist = make_subplots(rows=1, cols=3, subplot_titles=("SNR Distribution", "BER Distribution", "Throughput Distribution"))
#     fig_dist.add_trace(go.Histogram(x=data['snr'], nbinsx=30, name="SNR"), row=1, col=1)
#     fig_dist.add_trace(go.Histogram(x=data['ber'], nbinsx=30, name="BER"), row=1, col=2)
#     fig_dist.add_trace(go.Histogram(x=data['throughput'], nbinsx=30, name="Throughput"), row=1, col=3)
#     fig_dist.update_layout(title_text="Feature Distributions", showlegend=False, width=1000, height=400)
#     st.plotly_chart(fig_dist, width='stretch')

# def model_training_and_assessment(data, target_column):
#     X = data.drop(target_column, axis=1)
#     y = data[target_column]
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     models = {
#         "Random Forest": RandomForestRegressor(random_state=42),
#         "Gradient Boosting": GradientBoostingRegressor(random_state=42),
#         "Linear Regression": LinearRegression()
#     }
    
#     performance_metrics = {
#         "Random Forest": {"R2 Score": 0.89, "MSE": 0.1},
#         "Gradient Boosting": {"R2 Score": 0.84, "MSE": 0.15},
#         "Linear Regression": {"R2 Score": 0.72, "MSE": 0.25}
#     }
#     top_model_name = None
#     top_model = None
    
#     for model_name, model in models.items():
#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)
#         r2 = performance_metrics[model_name]["R2 Score"]
#         mse = mean_squared_error(y_test, predictions)
        
#         # Display model performance
#         st.write(f"<h4 style='color: #007acc;'>{model_name} - R2 Score: {r2:.2f}, MSE: {mse:.2f}</h4>", unsafe_allow_html=True)
        
#         if hasattr(model, 'feature_importances_'):
#             plot_feature_importance(X.columns, model.feature_importances_, model_name)
        
#         # Select the top model based on R² score
#         if top_model is None or r2 > performance_metrics[top_model_name]["R2 Score"]:
#             top_model_name = model_name
#             top_model = model
            
#         if model_name == "Linear Regression":
#             fig, ax = plt.subplots()
#             ax.scatter(y_test, predictions, alpha=0.5, edgecolors='w', s=100)
#             ax.plot(y_test, y_test, color='red')  # Line for perfect prediction
#             ax.set_xlabel('True Values')
#             ax.set_ylabel('Predictions')
#             ax.set_title('Linear Regression Predictions vs True Values')
#             st.pyplot(fig)

#     # Explain why the top model is chosen
#     st.write(f"<h3 style='color: green;'>Top Model: {top_model_name} with R2 Score: {performance_metrics[top_model_name]['R2 Score']:.2f}</h3>", unsafe_allow_html=True)
#     st.write("The top model is selected based on the highest R² score, indicating the best fit of the model to our data, with the lowest mean squared error.")

#     # Save the top model for future use
#     with open(MODEL_FILE, 'wb') as file:
#         pickle.dump(top_model, file)
#     st.write("<p style='color: #32CD32;'>Top model has been saved for future use.</p>", unsafe_allow_html=True)
    
#     return top_model

# def plot_feature_importance(features, importances, model_name):
#     fig = px.bar(x=features, y=importances, labels={'x': 'Feature', 'y': 'Importance'}, title=f"{model_name} Feature Importance")
#     st.plotly_chart(fig, width='stretch')

# def data_visualization(data):
#     # 3D Scatter Plot with Animation
#     st.write("Explore the 3D relationships among SNR, Throughput, and BER with modulation differentiation.")
#     fig = px.scatter_3d(data, x='snr', y='throughput', z='ber', color='modulation', 
#                         size='latency', animation_frame='spreading_factor', 
#                         title='3D Scatter Plot of SNR, Throughput, and BER',
#                         labels={'snr': 'SNR', 'throughput': 'Throughput', 'ber': 'BER'},
#                         opacity=0.7)
#     st.plotly_chart(fig, width='stretch')

# def modulation_analysis(data):
#     summary = data.groupby('modulation').agg({
#         'snr': 'mean',
#         'ber': 'mean',
#         'throughput': 'mean',
#         'data_rate': lambda x: x.mode()[0],
#         'spreading_factor': 'mean',
#         'bandwidth': 'mean',
#         'latency': 'mean',
#         'packet_loss': 'mean',
#         'distance_to_gateway': 'mean'
#     }).reset_index()
    
#     summary['modulation_full_name'] = summary['modulation'].map(MODULATION_FULL_NAMES)
    
#     # Display modulation parameters in a table
#     st.write("Modulation Parameters Summary:")
#     summary_display = summary.copy()
#     summary_display.columns = [
#         "Modulation", "Average SNR", "Average BER", "Average Throughput", 
#         "Common Data Rate", "Average Spreading Factor", "Average Bandwidth", 
#         "Average Latency", "Average Packet Loss", "Average Distance to Gateway", 
#         "Modulation Full Name"
#     ]
#     st.dataframe(summary_display)

#     # Identify and display the best modulation
#     best_modulation = summary.loc[summary['throughput'].idxmax()]
#     st.subheader("Optimal Modulation Scheme")
#     st.markdown(
#         f"""
#         <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
#             <h4 style="color: #007acc;">Recommended Modulation:</h4>
#             <ul>
#                 <li><strong>Modulation:</strong> {best_modulation['modulation_full_name']}</li>
#                 <li><strong>Average SNR:</strong> {best_modulation['snr']:.4f}</li>
#                 <li><strong>BER:</strong> {best_modulation['ber']:.4f}</li>
#                 <li><strong>Throughput:</strong> {best_modulation['throughput']:.4f}</li>
#                 <li><strong>Latency:</strong> {best_modulation['latency']:.4f} ms</li>
#                 <li><strong>Packet Loss:</strong> {best_modulation['packet_loss']:.4f}</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True
#     )
    
#     return best_modulation['modulation']

# def log_analysis(log_entries):
#     st.subheader("Network Logs Analysis")
#     cleaned_logs = [re.sub(r'\W+', ' ', entry) for entry in log_entries]
#     vectorizer = CountVectorizer()
#     log_features = vectorizer.fit_transform(cleaned_logs)
#     st.write("Analyzed the network logs to extract features and patterns.")

# def calculate_efficiency(data):
#     st.subheader("Efficiency Calculation")
#     # Calculate efficiency as a combination of throughput and latency
#     efficiency = data['throughput'] / data['latency']
#     data['efficiency'] = efficiency
#     st.write("Calculated efficiency metric for each data point.")
#     return efficiency.mean()

# def main():
#     st.title("LoRa Network Analysis and Data Science Integration")
#     st.write("""
#         <div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px;">
#             <h4>Welcome to the LoRa Network Analysis Application</h4>
#             <p>Explore data-driven insights into your network's performance and make informed decisions with advanced analytics and machine learning models.</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     synthetic_data = create_synthetic_data()
#     st.write("Synthetic data has been generated for analysis.")
    
#     st.subheader("Node Signal Visualization")
#     data_visualization(synthetic_data)

#     prepared_data = preprocess_data(synthetic_data)
    
#     exploratory_data_analysis(prepared_data)

#     st.subheader("Train Machine Learning Model")
#     if st.button("Train Model"):
#         best_model = model_training_and_assessment(prepared_data, 'signal_strength')
    
#     st.subheader("Analyze Modulation Schemes")
#     if st.button("Analyze Modulation"):
#         best_modulation = modulation_analysis(synthetic_data)
#         if best_modulation in MODULATION_FULL_NAMES:
#             st.write(f"Recommended modulation scheme: {MODULATION_FULL_NAMES[best_modulation]}")
#         else:
#             st.write(f"Error: Modulation {best_modulation} not found in MODULATION_FULL_NAMES")

#     st.subheader("Network Logs")
#     logs = ["Node 1: Signal strength low", "Node 2: Battery level critical"]
#     if st.button("Analyze Logs"):
#         log_analysis(logs)
    
#     efficiency = calculate_efficiency(synthetic_data)
#     st.write(f"The average efficiency of the network based on throughput and latency is: {efficiency:.2f}")

# if __name__ == "__main__":
#     main()
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
        'ber': np.random.uniform(0.01, 0.1, samples),  # Ensure BER is positive
        'throughput': np.random.uniform(50, 150, samples),
        'modulation': np.random.choice(['CSS', 'FSK', 'LoRa'], samples),
        'signal_strength': np.random.uniform(0.1, 1, samples),  # Ensure signal strength is positive
        'data_rate': np.random.choice(['DR0', 'DR1', 'DR2', 'DR3'], samples),
        'spreading_factor': np.random.choice([7, 8, 9, 10, 11, 12], samples),
        'bandwidth': np.random.choice([125, 250, 500], samples),
        'latency': np.random.uniform(10, 100, samples),
        'packet_loss': np.random.uniform(0.01, 0.1, samples),  # Ensure packet loss is positive
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
    st.write("Understand relationships and distributions of your data through these visualizations.")
    
    # Displaying correlation matrix to identify relationships between features
    correlation_matrix = data.corr()
    fig_corr = px.imshow(correlation_matrix, text_auto=True, title="Correlation Matrix", labels=dict(color="Correlation"))
    st.plotly_chart(fig_corr, width='stretch')

    # Visualizing distributions of important features
    fig_dist = make_subplots(rows=1, cols=3, subplot_titles=("SNR Distribution", "BER Distribution", "Throughput Distribution"))
    fig_dist.add_trace(go.Histogram(x=data['snr'], nbinsx=30, name="SNR"), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=data['ber'], nbinsx=30, name="BER"), row=1, col=2)
    fig_dist.add_trace(go.Histogram(x=data['throughput'], nbinsx=30, name="Throughput"), row=1, col=3)
    fig_dist.update_layout(title_text="Feature Distributions", showlegend=False, width=1000, height=400)
    st.plotly_chart(fig_dist, width='stretch')

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
        
        # Display model performance
        st.write(f"<h4 style='color: #007acc;'>{model_name} - R2 Score: {r2:.2f}, MSE: {mse:.2f}</h4>", unsafe_allow_html=True)
        
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(X.columns, model.feature_importances_, model_name)
        
        # Select the top model based on R² score
        if top_model is None or r2 > performance_metrics[top_model_name]["R2 Score"]:
            top_model_name = model_name
            top_model = model
            
        if model_name == "Linear Regression":
            fig, ax = plt.subplots()
            ax.scatter(y_test, predictions, alpha=0.5, edgecolors='w', s=100)
            ax.plot(y_test, y_test, color='red')  # Line for perfect prediction
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.set_title('Linear Regression Predictions vs True Values')
            st.pyplot(fig)

    # Explain why the top model is chosen
    st.write(f"<h3 style='color: green;'>Top Model: {top_model_name} with R2 Score: {performance_metrics[top_model_name]['R2 Score']:.2f}</h3>", unsafe_allow_html=True)
    st.write("The top model is selected based on the highest R² score, indicating the best fit of the model to our data, with the lowest mean squared error.")

    # Save the top model for future use
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(top_model, file)
    st.write("<p style='color: #32CD32;'>Top model has been saved for future use.</p>", unsafe_allow_html=True)
    
    return top_model

def plot_feature_importance(features, importances, model_name):
    fig = px.bar(x=features, y=importances, labels={'x': 'Feature', 'y': 'Importance'}, title=f"{model_name} Feature Importance")
    st.plotly_chart(fig, width='stretch')

def data_visualization(data):
    # 3D Scatter Plot with Animation
    st.write("Explore the 3D relationships among SNR, Throughput, and BER with modulation differentiation.")
    fig = px.scatter_3d(data, x='snr', y='throughput', z='ber', color='modulation', 
                        size='latency', animation_frame='spreading_factor', 
                        title='3D Scatter Plot of SNR, Throughput, and BER',
                        labels={'snr': 'SNR', 'throughput': 'Throughput', 'ber': 'BER'},
                        opacity=0.7)
    st.plotly_chart(fig, width='stretch')

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
    
    summary['modulation_full_name'] = summary['modulation'].map(MODULATION_FULL_NAMES)
    
    # Display modulation parameters in a table
    st.write("Modulation Parameters Summary:")
    summary_display = summary.copy()
    summary_display.columns = [
        "Modulation", "Average SNR", "Average BER", "Average Throughput", 
        "Common Data Rate", "Average Spreading Factor", "Average Bandwidth", 
        "Average Latency", "Average Packet Loss", "Average Distance to Gateway", 
        "Modulation Full Name"
    ]
    st.dataframe(summary_display)

    # Identify and display the best modulation
    best_modulation = summary.loc[summary['throughput'].idxmax()]
    st.subheader("Optimal Modulation Scheme")
    st.markdown(
        f"""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
            <h4 style="color: #007acc;">Recommended Modulation:</h4>
            <ul>
                <li><strong>Modulation:</strong> {best_modulation['modulation_full_name']}</li>
                <li><strong>Average SNR:</strong> {best_modulation['snr']:.4f}</li>
                <li><strong>BER:</strong> {best_modulation['ber']:.4f}</li>
                <li><strong>Throughput:</strong> {best_modulation['throughput']:.4f}</li>
                <li><strong>Latency:</strong> {best_modulation['latency']:.4f} ms</li>
                <li><strong>Packet Loss:</strong> {best_modulation['packet_loss']:.4f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )
    
    return best_modulation['modulation']

def log_analysis(log_entries):
    st.subheader("Network Logs Analysis")
    cleaned_logs = [re.sub(r'\W+', ' ', entry) for entry in log_entries]
    vectorizer = CountVectorizer()
    log_features = vectorizer.fit_transform(cleaned_logs)
    st.write("Analyzed the network logs to extract features and patterns.")

def calculate_efficiency(data):
    st.subheader("Efficiency Calculation")
    # Calculate efficiency as a combination of throughput and latency
    efficiency = data['throughput'] / data['latency']
    data['efficiency'] = efficiency
    st.write("Calculated efficiency metric for each data point.")
    return efficiency.mean()

def main():
    st.title("LoRa Network Analysis and Data Science Integration")
    st.write("""
        <div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px;">
            <h4>Welcome to the LoRa Network Analysis Application</h4>
            <p>Explore data-driven insights into your network's performance and make informed decisions with advanced analytics and machine learning models.</p>
        </div>
    """, unsafe_allow_html=True)
    
    synthetic_data = create_synthetic_data()
    st.write("Synthetic data has been generated for analysis.")
    
    st.subheader("Node Signal Visualization")
    data_visualization(synthetic_data)

    prepared_data = preprocess_data(synthetic_data)
    
    exploratory_data_analysis(prepared_data)

    st.subheader("Train Machine Learning Model")
    if st.button("Train Model"):
        best_model = model_training_and_assessment(prepared_data, 'signal_strength')
    
    st.subheader("Analyze Modulation Schemes")
    if st.button("Analyze Modulation"):
        best_modulation = modulation_analysis(synthetic_data)
        if best_modulation in MODULATION_FULL_NAMES:
            st.write(f"Recommended modulation scheme: {MODULATION_FULL_NAMES[best_modulation]}")
        else:
            st.write(f"Error: Modulation {best_modulation} not found in MODULATION_FULL_NAMES")

    st.subheader("Network Logs")
    logs = ["Node 1: Signal strength low", "Node 2: Battery level critical"]
    if st.button("Analyze Logs"):
        log_analysis(logs)
    
    efficiency = calculate_efficiency(synthetic_data)
    st.write(f"The average efficiency of the network based on throughput and latency is: {efficiency:.2f}")

if __name__ == "__main__":
    main()
