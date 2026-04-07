import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def generate_sample_data():
    """Generate sample UPI transaction data for training"""
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 10000
    
    data = {
        'amount': np.random.lognormal(7, 1.5, n_samples).clip(1, 1000000),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'sender_frequency': np.random.exponential(5, n_samples),
        'receiver_frequency': np.random.exponential(3, n_samples),
        'location_mismatch': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'device_change': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'amount_deviation': np.random.exponential(2, n_samples),
        'time_since_last': np.random.exponential(60, n_samples), # minutes
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'transaction_type': np.random.choice(['p2p', 'p2m', 'bill_payment'], n_samples, p=[0.6, 0.3, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Create fraud labels based on patterns
    fraud_conditions = (
        (df['amount'] > 50000) |
        (df['location_mismatch'] == 1) |
        (df['device_change'] == 1) |
        (df['amount_deviation'] > 5) |
        (df['hour_of_day'] < 6) |
        (df['sender_frequency'] < 0.5)
    )
    
    # Add some randomness to make it more realistic
    fraud_probability = fraud_conditions.astype(int) * 0.7 + np.random.random(n_samples) * 0.3
    df['is_fraud'] = (fraud_probability > 0.5).astype(int)
    
    return df

def preprocess_data(df, scaler=None, label_encoders=None, is_training=True):
    """Preprocess the data for ML model"""
    
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Handle categorical variables
    if is_training:
        label_encoders = {}
        
        # Encode transaction_type
        le_transaction = LabelEncoder()
        df_processed['transaction_type_encoded'] = le_transaction.fit_transform(df_processed['transaction_type'])
        label_encoders['transaction_type'] = le_transaction
        
    else:
        # Use existing encoders for prediction
        df_processed['transaction_type_encoded'] = label_encoders['transaction_type'].transform(df_processed['transaction_type'])
    
    # Select features for training
    feature_columns = [
        'amount', 'hour_of_day', 'day_of_week', 'sender_frequency',
        'receiver_frequency', 'location_mismatch', 'device_change',
        'amount_deviation', 'time_since_last', 'is_weekend',
        'transaction_type_encoded'
    ]
    
    X = df_processed[feature_columns]
    
    # Handle target variable
    if 'is_fraud' in df_processed.columns:
        y = df_processed['is_fraud']
    else:
        y = None
    
    # Scale numerical features
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, y, scaler, label_encoders

def train_model():
    """Train the fraud detection model"""
    
    print("Generating sample data...")
    df = generate_sample_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud distribution: {df['is_fraud'].value_counts()}")
    
    # Preprocess data
    X, y, scaler, label_encoders = preprocess_data(df, is_training=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_names = [
        'amount', 'hour_of_day', 'day_of_week', 'sender_frequency',
        'receiver_frequency', 'location_mismatch', 'device_change',
        'amount_deviation', 'time_since_last', 'is_weekend',
        'transaction_type_encoded'
    ]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and preprocessing components
    joblib.dump(model, 'models/fraud_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    print("\nModel and preprocessing components saved successfully!")
    
    return model, scaler, label_encoders

def load_model():
    """Load the trained model and preprocessing components"""
    try:
        model = joblib.load('models/fraud_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        return model, scaler, label_encoders
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        return None, None, None

if __name__ == "__main__":
    # Train the model
    model, scaler, label_encoders = train_model()
    
    # Test loading
    print("\nTesting model loading...")
    loaded_model, loaded_scaler, loaded_encoders = load_model()
    
    if loaded_model is not None:
        print("Model loaded successfully!")
    else:
        print("Failed to load model.")
