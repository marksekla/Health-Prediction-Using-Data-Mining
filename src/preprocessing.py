import pandas as pd
import numpy as np

def load_data(file_path):
    """Load dataset from CSV file"""
    try:
        # Read CSV
        data = pd.read_csv(file_path)
        print(f"Loaded dataset: {data.shape[0]} patients, {data.shape[1]} features")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_heart_data(data):
    """Clean and prepare heart disease data"""
    # Check for missing values
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(f"Found {missing.sum()} missing values")
        # Drop rows with missing values
        data = data.dropna()
        print(f"Dataset after cleaning: {data.shape[0]} patients")
    
    # Create age groups
    data['age_bin'] = pd.cut(data['age'], bins=[0, 40, 50, 60, 100], 
                            labels=['young', 'middle_aged', 'senior', 'elderly'])
    
    # Create blood pressure groups
    data['trestbps_bin'] = pd.cut(data['trestbps'], bins=[0, 120, 140, 180, 300], 
                                 labels=['normal', 'elevated', 'high', 'very_high'])
    
    # Create cholesterol groups
    data['chol_bin'] = pd.cut(data['chol'], bins=[0, 200, 240, 300, 600], 
                             labels=['desirable', 'borderline', 'high', 'very_high'])
    
    # Make sure target is binary
    if 'target' in data.columns:
        data['heart_disease'] = data['target']
    elif 'num' in data.columns:
        data['heart_disease'] = (data['num'] > 0).astype(int)
    
    # Select categorical features for Apriori
    categorical_features = [col for col in data.columns if col.endswith('_bin')] + ['sex', 'cp', 'heart_disease']
    
    # Convert to category type
    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].astype('category')
    
    print(f"Preprocessing complete with {len(categorical_features)} features")
    return data[categorical_features]

def create_transaction_data(data):
    """Convert data to transaction format for Apriori"""
    transactions = []
    for _, row in data.iterrows():
        # Create feature=value pairs
        items = [f"{col}={val}" for col, val in row.items() if pd.notna(val)]
        transactions.append(items)
    
    print(f"Created {len(transactions)} patient transactions")
    return transactions