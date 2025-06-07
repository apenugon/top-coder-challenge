import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load and prepare the public cases data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    rows = []
    for case in data:
        input_data = case['input']
        rows.append({
            'trip_duration_days': input_data['trip_duration_days'],
            'miles_traveled': input_data['miles_traveled'],
            'total_receipts_amount': input_data['total_receipts_amount'],
            'reimbursement': case['expected_output']
        })
    
    return pd.DataFrame(rows)

def create_basic_features(df):
    """Create basic features from the raw data"""
    df = df.copy()
    
    # Basic derived features
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    
    # Per diem calculation (assume base rate of ~$100/day)
    df['base_per_diem'] = df['trip_duration_days'] * 100
    df['reimbursement_ratio'] = df['reimbursement'] / df['base_per_diem']
    
    return df

def train_decision_tree(X, y, max_depth=10, min_samples_split=20, min_samples_leaf=10):
    """Train a decision tree regressor"""
    
    # Split the data: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of total
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train the decision tree
    dt = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    dt.fit(X_train, y_train)
    
    # Make predictions
    train_pred = dt.predict(X_train)
    val_pred = dt.predict(X_val)
    test_pred = dt.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print("\n=== Model Performance ===")
    print(f"Training   - RMSE: {train_rmse:.2f}, R²: {train_r2:.3f}, MAE: {train_mae:.2f}")
    print(f"Validation - RMSE: {val_rmse:.2f}, R²: {val_r2:.3f}, MAE: {val_mae:.2f}")
    print(f"Test       - RMSE: {test_rmse:.2f}, R²: {test_r2:.3f}, MAE: {test_mae:.2f}")
    
    return dt, (X_train, X_val, X_test), (y_train, y_val, y_test), (train_pred, val_pred, test_pred)

def analyze_feature_importance(dt, feature_names):
    """Analyze and display feature importance"""
    importance = dt.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n=== Feature Importance ===")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")
    
    return feature_importance

def print_decision_rules(dt, feature_names, max_depth=3):
    """Print the decision tree rules in readable format"""
    print(f"\n=== Decision Tree Rules (max depth {max_depth} shown) ===")
    tree_rules = export_text(dt, feature_names=feature_names, max_depth=max_depth)
    print(tree_rules)

def analyze_predictions(y_true, y_pred, dataset_name="Dataset"):
    """Analyze prediction patterns"""
    residuals = y_true - y_pred
    
    print(f"\n=== {dataset_name} Prediction Analysis ===")
    print(f"Mean residual: {np.mean(residuals):.2f}")
    print(f"Std residual: {np.std(residuals):.2f}")
    print(f"Max underestimate: {np.min(residuals):.2f}")
    print(f"Max overestimate: {np.max(residuals):.2f}")
    
    # Find cases with largest errors
    abs_residuals = np.abs(residuals)
    worst_idx = np.argmax(abs_residuals)
    
    print(f"Worst prediction error: {abs_residuals[worst_idx]:.2f}")
    print(f"True value: {y_true.iloc[worst_idx]:.2f}, Predicted: {y_pred[worst_idx]:.2f}")

def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} cases")
    
    print("\n=== Data Overview ===")
    print(df.describe())
    
    # Create features
    df_features = create_basic_features(df)
    
    # Define features for modeling
    feature_columns = [
        'trip_duration_days', 
        'miles_traveled', 
        'total_receipts_amount',
        'miles_per_day', 
        'receipts_per_day'
    ]
    
    X = df_features[feature_columns]
    y = df_features['reimbursement']
    
    print(f"\n=== Feature Statistics ===")
    print(X.describe())
    
    # Train decision tree
    print("\n=== Training Decision Tree ===")
    dt, (X_train, X_val, X_test), (y_train, y_val, y_test), (train_pred, val_pred, test_pred) = train_decision_tree(X, y)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(dt, feature_columns)
    
    # Print decision rules
    print_decision_rules(dt, feature_columns)
    
    # Analyze predictions
    analyze_predictions(y_val, val_pred, "Validation")
    analyze_predictions(y_test, test_pred, "Test")
    
    # Test different tree depths
    print("\n=== Testing Different Tree Depths ===")
    depths = [3, 5, 7, 10, 15, 20]
    results = []
    
    for depth in depths:
        dt_temp = DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt_temp.fit(X_train, y_train)
        val_pred_temp = dt_temp.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred_temp))
        val_r2 = r2_score(y_val, val_pred_temp)
        results.append((depth, val_rmse, val_r2))
        print(f"Depth {depth}: RMSE={val_rmse:.2f}, R²={val_r2:.3f}")
    
    # Find best depth
    best_depth = min(results, key=lambda x: x[1])[0]
    print(f"\nBest tree depth: {best_depth}")
    
    return dt, df_features, feature_importance

if __name__ == "__main__":
    model, data, importance = main() 