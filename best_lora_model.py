import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import os

# Constants
MODEL_FILE = r'D:\Internship\best_lora_model.pkl'

def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic data for LoRa network analysis.
    """
    np.random.seed(42)
    data = {
        'node': np.random.choice(['Node1', 'Node2', 'Node3', 'Node4'], num_samples),
        'snr': np.random.uniform(5, 20, num_samples),
        'ber': np.random.uniform(0, 0.1, num_samples),
        'throughput': np.random.uniform(50, 150, num_samples),
        'modulation': np.random.choice(['CSS', 'FSK', 'LoRa'], num_samples),
        'signal_strength': np.random.uniform(0, 1, num_samples),
        'data_rate': np.random.choice(['DR0', 'DR1', 'DR2', 'DR3'], num_samples),
        'spreading_factor': np.random.choice([7, 8, 9, 10, 11, 12], num_samples),
        'bandwidth': np.random.choice([125, 250, 500], num_samples)
    }
    return pd.DataFrame(data)

def preprocess_data(data):
    """
    Data cleaning, transformation, and scaling.
    """
    numerical_features = data.select_dtypes(include=[np.number])
    data[numerical_features.columns] = numerical_features.fillna(numerical_features.mean())
    scaler = StandardScaler()
    data[numerical_features.columns] = scaler.fit_transform(numerical_features)
    
    categorical_features = data.select_dtypes(include=[object])
    for col in categorical_features.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    return data

def train_and_save_best_model(data, target):
    """
    Train multiple regression models and save the best model.
    """
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
    
    best_score = -np.inf
    best_model = None
    
    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grid[name], cv=3, scoring='r2')
        grid_search.fit(X_train, y_train)
        current_best_model = grid_search.best_estimator_
        predictions = current_best_model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        if r2 > best_score:
            best_score = r2
            best_model = current_best_model
            
    # Save the best model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model saved with R2 Score: {best_score:.2f}")

if __name__ == "__main__":
    # Generate synthetic data
    nodes_df = generate_synthetic_data()
    nodes_df = preprocess_data(nodes_df)
    
    # Train and save the best model
    train_and_save_best_model(nodes_df, 'signal_strength')

