import pandas as pd
import numpy as np

def generate_node_dataset(number_of_nodes=100):
    # Craft dataset for nodes with relevant attributes
    node_attributes = {
        'node_id': [f'Node_{i}' for i in range(number_of_nodes)],
        'x_coordinate': np.random.uniform(0, 100, number_of_nodes),  # X-coordinate in meters
        'y_coordinate': np.random.uniform(0, 100, number_of_nodes),  # Y-coordinate in meters
        'altitude': np.random.uniform(0, 20, number_of_nodes),  # Altitude in meters
        'battery_percentage': np.random.uniform(50, 100, number_of_nodes),  # Battery percentage
        'operational_status': np.random.choice(['active', 'inactive'], number_of_nodes, p=[0.9, 0.1])  # Status of the node
    }
    node_dataframe = pd.DataFrame(node_attributes)
    # Save the node data to an XLSX file
    node_dataframe.to_excel('node_data.xlsx', index=False)
    return node_dataframe

def create_distance_dataset(number_of_nodes=100):
    # Develop dataset containing distances between nodes
    distance_data = {
        'distance_id': [],
        'first_node': [],
        'second_node': [],
        'distance_value': []
    }
    for i in range(number_of_nodes):
        for j in range(i + 1, number_of_nodes):
            distance_data['distance_id'].append(f'Distance_{i}_{j}')
            distance_data['first_node'].append(f'Node_{i}')
            distance_data['second_node'].append(f'Node_{j}')
            distance_data['distance_value'].append(np.random.uniform(0, 100))  # Distance in meters
    distance_dataframe = pd.DataFrame(distance_data)
    # Save the distance data to an XLSX file
    distance_dataframe.to_excel('distance_data.xlsx', index=False)
    return distance_dataframe

def create_obstacle_dataset(number_of_obstacles=50):
    # Create dataset for obstacles with position and type
    obstacle_data = {
        'obstacle_id': [f'Obstacle_{i}' for i in range(number_of_obstacles)],
        'x_position': np.random.uniform(0, 100, number_of_obstacles),
        'y_position': np.random.uniform(0, 100, number_of_obstacles),
        'height': np.random.uniform(1, 10, number_of_obstacles),  # Height in meters
        'material': np.random.choice(['wood', 'metal', 'concrete'], number_of_obstacles)
    }
    obstacle_dataframe = pd.DataFrame(obstacle_data)
    # Save the obstacle data to an XLSX file
    obstacle_dataframe.to_excel('obstacle_data.xlsx', index=False)
    return obstacle_dataframe

def create_environmental_dataset(number_of_records=100):
    # Create dataset for environmental factors like pressure, humidity, and temperature
    environmental_data = {
        'record_id': [f'Record_{i}' for i in range(number_of_records)],
        'pressure': np.random.uniform(950, 1050, number_of_records),  # Pressure in hPa
        'humidity': np.random.uniform(30, 100, number_of_records),  # Humidity in percentage
        'temperature': np.random.uniform(-10, 40, number_of_records)  # Temperature in Celsius
    }
    environmental_dataframe = pd.DataFrame(environmental_data)
    # Save the environmental data to an XLSX file
    environmental_dataframe.to_excel('environmental_data.xlsx', index=False)
    return environmental_dataframe

# Generate and save all datasets
node_data = generate_node_dataset()
distance_data = create_distance_dataset()
obstacle_data = create_obstacle_dataset()
environmental_data = create_environmental_dataset()
