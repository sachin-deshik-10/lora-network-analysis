import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import streamlit as st
import plotly.express as px
from dask import dataframe as dd
import re
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_data(data, missing_strategy="mean", scaler_type="standard"):
    """
    Preprocess data by handling missing values and scaling numeric features.
    """
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
    for col in categorical_features.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    return data

def train_model_regression(data, target, n_estimators=100, max_depth=None):
    """
    Train a Random Forest regression model.
    """
    X = data.drop(target, axis=1)
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = r2_score(y_test, predictions)
    
    return model, accuracy

def visualize_data(data, x="x", y="y", color="signal_strength", title="Node Signal Strength", theme="plotly"):
    """
    Visualize data using a scatter plot.
    """
    fig = px.scatter(data, x=x, y=y, color=color, title=title, template=theme)
    st.plotly_chart(fig)

def process_large_data(file_path, file_format="csv"):
    """
    Process large datasets using Dask.
    """
    try:
        if file_format == "csv":
            df = dd.read_csv(file_path)
        elif file_format == "parquet":
            df = dd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        result = df.groupby('modulation').mean().compute()
        return result
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return None

def analyze_logs(logs, ngram_range=(1, 1)):
    """
    Analyze logs using text vectorization.
    """
    logs = [re.sub(r'\W+', ' ', log) for log in logs]
    
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(logs)
    
    return X

def retrieve_data():
    """
    Retrieve datasets from specified paths.
    """
    try:
        nodes_df = pd.read_csv(r"D:\Internship\nodes.csv")
        distances_df = pd.read_csv(r"D:\Internship\distances.csv")
        obstacles_df = pd.read_csv(r"D:\Internship\obstacles.csv")
        pressure_df = pd.read_csv(r"D:\Internship\pressure.csv")
        humidity_df = pd.read_csv(r"D:\Internship\humidity.csv")
        temperature_df = pd.read_csv(r"D:\Internship\temperature.csv")
        
        return nodes_df, distances_df, obstacles_df, pressure_df, humidity_df, temperature_df
    except FileNotFoundError as e:
        st.error(f"Data file missing: {e}")
        return None

def main():
    st.title(" LoRa Network Analysis")
    
    data = retrieve_data()
    if data:
        nodes_df, distances_df, obstacles_df, pressure_df, humidity_df, temperature_df = data
        nodes_df = preprocess_data(nodes_df)
        
        st.subheader("Node Locations")
        visualize_data(nodes_df)
        
        st.subheader("Train Machine Learning Model")
        if st.button("Train Model"):
            model, accuracy = train_model_regression(nodes_df, 'signal_strength')
            st.write(f"Model R^2 Score: {accuracy:.2f}")
        
        st.subheader("Process Large Data with Dask")
        if st.button("Process Data"):
            result = process_large_data(r"D:\Internship\large_dataset.csv")
            if result is not None:
                st.write(result)
        
        st.subheader("Analyze Network Logs")
        logs = ["Node 1: Signal strength low", "Node 2: Battery level critical"]
        if st.button("Analyze Logs"):
            log_analysis = analyze_logs(logs)
            st.write("Log Analysis Complete")

if __name__ == "__main__":
    main()
